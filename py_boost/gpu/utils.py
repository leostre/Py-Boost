"""Utilities used for trees building and inference"""

import hashlib
import math

import numba
import numpy as np

from py_boost.multioutput.split_diagnostics import action_gain_regret, normalized_gain_regret
from time import perf_counter

try:
    import cupy as cp

    CUDA_FOUND = True
except Exception:
    CUDA_FOUND = False


_DETERMINISTIC_HISTOGRAM_PROFILE = None
_DETERMINISTIC_HISTOGRAM_LAYOUT_COUNTS = None


def start_deterministic_histogram_profile():
    """Start collecting synchronized timings for deterministic histograms."""

    global _DETERMINISTIC_HISTOGRAM_PROFILE
    if _DETERMINISTIC_HISTOGRAM_PROFILE is not None:
        raise RuntimeError('deterministic histogram profiling is already active')
    _DETERMINISTIC_HISTOGRAM_PROFILE = []


def stop_deterministic_histogram_profile():
    """Stop profiling and return the collected call records."""

    global _DETERMINISTIC_HISTOGRAM_PROFILE
    if _DETERMINISTIC_HISTOGRAM_PROFILE is None:
        raise RuntimeError('deterministic histogram profiling is not active')
    records = _DETERMINISTIC_HISTOGRAM_PROFILE
    _DETERMINISTIC_HISTOGRAM_PROFILE = None
    return records


def start_deterministic_histogram_layout_trace():
    """Start a lightweight count of selected layouts and node counts."""

    global _DETERMINISTIC_HISTOGRAM_LAYOUT_COUNTS
    _DETERMINISTIC_HISTOGRAM_LAYOUT_COUNTS = {}


def stop_deterministic_histogram_layout_trace():
    """Stop layout tracing and return a copy of the accumulated counts."""

    global _DETERMINISTIC_HISTOGRAM_LAYOUT_COUNTS
    counts = dict(_DETERMINISTIC_HISTOGRAM_LAYOUT_COUNTS or {})
    _DETERMINISTIC_HISTOGRAM_LAYOUT_COUNTS = None
    return counts


def validate_input_single(X, y, sample_weight):
    """Format single dataset to fit py_boost

    Args:
        X: np.ndarray, features
        y: np.ndarray, targets
        sample_weight: np.ndarray or None, sample weights

    Returns:
        tuple
    """
    y = np.asarray(y)
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    X = np.asarray(X)
    if np.issubdtype(X.dtype, np.integer):
        X = X.astype(np.float32)

    if (sample_weight is not None) and (len(sample_weight.shape) == 1):
        sample_weight = sample_weight[:, np.newaxis]

    return X, y, sample_weight


def validate_input(X, y, sample_weight=None, eval_sets=None):
    """Format train and valid datasets to fit py_boost

    Args:
        X: np.ndarray, features
        y: np.ndarray, targets
        sample_weight: np.ndarray or None, sample weights
        eval_sets: list of dicts, valid datasets

    Returns:
        tuple
    """
    X, y, sample_weight = validate_input_single(X, y, sample_weight)

    if eval_sets is None:
        eval_sets = []
    else:
        eval_sets = list(eval_sets)

    for val_arr in eval_sets:
        if 'sample_weight' not in val_arr:
            val_arr['sample_weight'] = None

        val_arr['X'], val_arr['y'], val_arr['sample_weight'] = validate_input_single(
            val_arr['X'], val_arr['y'], val_arr['sample_weight'])

    return X, y, sample_weight, eval_sets


def pad_and_move(arr, pad_size=4):
    """Pad array memory placeholder to make feature size divisible by pad_size and move to GPU.
    Returned array is the same size as input, but it is not contiguous.
    Basic memory placeholder size is divisible by pad_size and could be accesses via arr.base.
    This trick helps to speed up histogram computation for the large arrays.

    Args:
        arr: 2d np.ndarray of uint8 containing quantized features matrix
        pad_size: int, memory placeholder feature size will be divisible by pad_size

    Returns:
        cp.ndarray of quantized features
    """
    assert arr.dtype == np.uint8, 'Array dtype should be unsigned 8bit int'

    pfeats = math.ceil(arr.shape[1] / pad_size) * pad_size
    arr_cpu_padded = np.zeros((arr.shape[0], pfeats), dtype=np.uint8)
    arr_cpu_padded[:, :arr.shape[1]] = arr
    arr_gpu_padded = cp.asarray(arr_cpu_padded)

    return arr_gpu_padded[:, :arr.shape[1]]


isin_code = """
int fnode;
int leaf = 0;
int minval = 0;
int maxval = l - 1;

int node_val = nodes[index];

while (true) {{

    leaf = minval + (maxval - minval) / 2;
    fnode = un[leaf];

    if (fnode == node_val) {{

        isin={0};
        return;

    }} else if (minval >= maxval) {{

        isin={1};
        return ;

    }} else if (fnode < node_val) {{

        minval = leaf + 1;

    }} else  {{

        maxval = leaf - 1;

    }} 
}}

"""

isin_kernel = cp.ElementwiseKernel(
    'Q index, raw E nodes, raw R un, uint64 l',
    'bool isin',

    isin_code.format('true', 'false'),

    'isin_kernel'
) if CUDA_FOUND else None

isin_pos_kernel = cp.ElementwiseKernel(
    'Q index, raw E nodes, raw R un, uint64 l',
    'int32 isin',

    isin_code.format('leaf', -1),

    'isin_pos_kernel'
) if CUDA_FOUND else None


def isin(a, b, index=None, return_pos=False):
    """Check if b contains elements of a. Custom function is faster than build in cupy's

    Args:
        a: cp.ndarray to apply
        b: cp.ndarray lookup array
        index: cp.ndarray indices of a to check. If None, check for all a
        return_pos: bool, if True, return position of a[i] in b, otherwise return bool array if a[i] is in b

    Returns:
        cp.ndarray of bool or int
    """
    if index is None:
        index = cp.arange(a.shape[0], dtype=cp.uint64)

    if return_pos:
        return isin_pos_kernel(index, a, b, b.shape[0])
    else:
        return isin_kernel(index, a, b, b.shape[0])


histogram_kernel_idx = cp.ElementwiseKernel(
    """
    uint64 i_, uint64 j_, uint64 k_, 
    uint64 kk,

    raw uint64 jj,
    raw bool padded_bool_indexer,

    raw float32 target, 
    raw T arr, 
    raw int32 nodes,

    uint64 hlen,
    uint64 flen, 
    uint64 length,
    uint64 feats,
    uint64 nout 
    """,
    'raw float32 hist',

    """
    unsigned int feat_4t = arr[i_ * feats + j_];
    int d;
    int j;
    int val;
    int pos;
    float *x_ptr;
    float y = target[i_ * nout + k_];

    for (d = 0; d < 4; d++) {

        pos = (i_ + d) % 4;

        if (padded_bool_indexer[j_ * 4 + pos]) {

            val = (feat_4t >> (8 * pos)) % 256;
            j = jj[j_ * 4 + pos];
            x_ptr = &hist[0] +  kk * hlen + nodes[i_] * flen + j * length + val;
            atomicAdd(x_ptr, y); 
        }
    }

    """,

    'histogram_kernel_idx') if CUDA_FOUND else None


deterministic_histogram_reference_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void deterministic_histogram_reference(
        const unsigned char* arr,
        const unsigned long long padded_features,
        const float* target,
        const unsigned long long target_outputs,
        const int* nodes,
        const unsigned long long* row_indexer,
        const unsigned long long* col_indexer,
        const unsigned long long* out_indexer,
        float* histogram,
        const unsigned long long nrows,
        const unsigned long long noutputs,
        const unsigned long long nnodes,
        const unsigned long long nfeatures,
        const unsigned long long nbins)
    {
        const unsigned int outputs_per_block = 2;
        const unsigned int nodes_per_chunk = 8;
        const unsigned int warps = 2;
        const unsigned int lane = threadIdx.x & 31;
        const unsigned int warp = threadIdx.x >> 5;
        const unsigned long long output_batches =
            (noutputs + outputs_per_block - 1) / outputs_per_block;
        const unsigned long long output_batch = blockIdx.x % output_batches;
        const unsigned long long feature_and_node = blockIdx.x / output_batches;
        const unsigned long long feature_position = feature_and_node % nfeatures;
        const unsigned long long node_chunk = feature_and_node / nfeatures;
        const unsigned long long output_base = output_batch * outputs_per_block;
        const unsigned long long node_base = node_chunk * nodes_per_chunk;

        extern __shared__ float partial[];
        __shared__ int scratch_node[64];
        __shared__ unsigned char scratch_bin[64];
        __shared__ float scratch_value[2][64];

        const unsigned long long partial_size =
            warps * nodes_per_chunk * outputs_per_block * nbins;
        for (unsigned long long index = threadIdx.x;
             index < partial_size;
             index += blockDim.x) {
            partial[index] = 0.0f;
        }
        __syncthreads();

        const unsigned long long feature = col_indexer[feature_position];
        for (unsigned long long base = 0;
             base < nrows;
             base += blockDim.x) {
            const unsigned long long position = base + threadIdx.x;
            const bool active = position < nrows;
            const unsigned long long row = active ? row_indexer[position] : 0;
            scratch_node[threadIdx.x] = active ? nodes[row] : -1;
            scratch_bin[threadIdx.x] =
                active ? arr[row * padded_features + feature] : 0;
            #pragma unroll
            for (unsigned int output_offset = 0;
                 output_offset < outputs_per_block;
                 ++output_offset) {
                const unsigned long long output_position = output_base + output_offset;
                scratch_value[output_offset][threadIdx.x] =
                    active && output_position < noutputs
                    ? target[row * target_outputs + out_indexer[output_position]]
                    : 0.0f;
            }
            __syncwarp();

            if (lane == 0) {
                const unsigned int warp_start = warp * 32;
                for (unsigned int source = 0; source < 32; ++source) {
                    const unsigned int scratch = warp_start + source;
                    const int node = scratch_node[scratch];
                    const unsigned int bin = scratch_bin[scratch];
                    if (node >= (int)node_base
                        && node < (int)(node_base + nodes_per_chunk)
                        && node < (int)nnodes
                        && bin < nbins) {
                        const unsigned int node_offset = node - node_base;
                        #pragma unroll
                        for (unsigned int output_offset = 0;
                             output_offset < outputs_per_block;
                             ++output_offset) {
                            if (output_base + output_offset < noutputs) {
                                const unsigned long long partial_index =
                                    (((warp * nodes_per_chunk + node_offset)
                                      * outputs_per_block + output_offset)
                                     * nbins + bin);
                                partial[partial_index] +=
                                    scratch_value[output_offset][scratch];
                            }
                        }
                    }
                }
            }
            __syncwarp();
        }
        __syncthreads();

        const unsigned long long combined_size =
            nodes_per_chunk * outputs_per_block * nbins;
        for (unsigned long long index = threadIdx.x;
             index < combined_size;
             index += blockDim.x) {
            const unsigned long long bin = index % nbins;
            const unsigned long long node_and_output = index / nbins;
            const unsigned long long output_offset =
                node_and_output % outputs_per_block;
            const unsigned long long node_offset =
                node_and_output / outputs_per_block;
            const unsigned long long node = node_base + node_offset;
            const unsigned long long output = output_base + output_offset;
            if (node < nnodes && output < noutputs) {
                const unsigned long long first =
                    ((node_offset * outputs_per_block + output_offset)
                     * nbins + bin);
                const unsigned long long second =
                    (((nodes_per_chunk + node_offset) * outputs_per_block
                      + output_offset) * nbins + bin);
                const unsigned long long destination =
                    (((output * nnodes + node) * nfeatures + feature_position)
                     * nbins + bin);
                histogram[destination] = partial[first] + partial[second];
            }
        }
    }
    ''',
    'deterministic_histogram_reference',
) if CUDA_FOUND else None


deterministic_histogram_perf1_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void deterministic_histogram_perf1(
        const unsigned char* arr,
        const unsigned long long padded_features,
        const float* target,
        const unsigned long long target_outputs,
        const int* nodes,
        const unsigned long long* row_indexer,
        const unsigned long long* col_indexer,
        const unsigned long long* out_indexer,
        float* histogram,
        const unsigned long long nrows,
        const unsigned long long noutputs,
        const unsigned long long nnodes,
        const unsigned long long nfeatures,
        const unsigned long long nbins)
    {
        const unsigned int outputs_per_block = 2;
        const unsigned int nodes_per_chunk = 8;
        const unsigned int partial_warps = 2;
        const unsigned int lane = threadIdx.x & 31;
        const unsigned int warp = threadIdx.x >> 5;
        const unsigned int output_offset = warp >> 1;
        const unsigned int partial_warp = warp & 1;
        const unsigned long long output_batches =
            (noutputs + outputs_per_block - 1) / outputs_per_block;
        const unsigned long long output_batch = blockIdx.x % output_batches;
        const unsigned long long feature_and_node = blockIdx.x / output_batches;
        const unsigned long long feature_position = feature_and_node % nfeatures;
        const unsigned long long node_chunk = feature_and_node / nfeatures;
        const unsigned long long output_base = output_batch * outputs_per_block;
        const unsigned long long output_position = output_base + output_offset;
        const unsigned long long node_base = node_chunk * nodes_per_chunk;

        extern __shared__ float partial[];
        const unsigned long long partial_size =
            partial_warps * nodes_per_chunk * outputs_per_block * nbins;
        for (unsigned long long index = threadIdx.x;
             index < partial_size;
             index += blockDim.x) {
            partial[index] = 0.0f;
        }
        __syncthreads();

        const unsigned long long feature = col_indexer[feature_position];
        for (unsigned long long base = 0; base < nrows; base += 64) {
            const unsigned long long position =
                base + partial_warp * 32 + lane;
            const bool active = position < nrows && output_position < noutputs;
            const unsigned long long row = active ? row_indexer[position] : 0;
            const int local_node = active ? nodes[row] : -1;
            const unsigned int local_bin =
                active ? arr[row * padded_features + feature] : 0;
            const float local_value = active
                ? target[row * target_outputs + out_indexer[output_position]]
                : 0.0f;

            #pragma unroll
            for (unsigned int source = 0; source < 32; ++source) {
                const int node = __shfl_sync(0xffffffff, local_node, source);
                const unsigned int bin =
                    __shfl_sync(0xffffffff, local_bin, source);
                const float value =
                    __shfl_sync(0xffffffff, local_value, source);
                if (lane == (bin & 31)
                    && node >= (int)node_base
                    && node < (int)(node_base + nodes_per_chunk)
                    && node < (int)nnodes
                    && bin < nbins
                    && output_position < noutputs) {
                    const unsigned int node_offset = node - node_base;
                    const unsigned long long partial_index =
                        (((partial_warp * nodes_per_chunk + node_offset)
                          * outputs_per_block + output_offset)
                         * nbins + bin);
                    partial[partial_index] += value;
                }
            }
        }
        __syncthreads();

        const unsigned long long combined_size =
            nodes_per_chunk * outputs_per_block * nbins;
        for (unsigned long long index = threadIdx.x;
             index < combined_size;
             index += blockDim.x) {
            const unsigned long long bin = index % nbins;
            const unsigned long long node_and_output = index / nbins;
            const unsigned long long combined_output_offset =
                node_and_output % outputs_per_block;
            const unsigned long long node_offset =
                node_and_output / outputs_per_block;
            const unsigned long long node = node_base + node_offset;
            const unsigned long long output =
                output_base + combined_output_offset;
            if (node < nnodes && output < noutputs) {
                const unsigned long long first =
                    ((node_offset * outputs_per_block + combined_output_offset)
                     * nbins + bin);
                const unsigned long long second =
                    (((nodes_per_chunk + node_offset) * outputs_per_block
                      + combined_output_offset) * nbins + bin);
                const unsigned long long destination =
                    (((output * nnodes + node) * nfeatures + feature_position)
                     * nbins + bin);
                histogram[destination] = partial[first] + partial[second];
            }
        }
    }
    ''',
    'deterministic_histogram_perf1',
) if CUDA_FOUND else None


deterministic_histogram_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void deterministic_histogram(
        const unsigned char* arr,
        const unsigned long long padded_features,
        const float* target,
        const unsigned long long target_outputs,
        const int* nodes,
        const unsigned long long* row_indexer,
        const unsigned long long* col_indexer,
        const unsigned long long* out_indexer,
        float* histogram,
        const unsigned long long nrows,
        const unsigned long long noutputs,
        const unsigned long long nnodes,
        const unsigned long long nfeatures,
        const unsigned long long nbins)
    {
        const unsigned int outputs_per_block = 1;
        const unsigned int nodes_per_chunk = 16;
        const unsigned int partial_warps = 2;
        const unsigned int lane = threadIdx.x & 31;
        const unsigned int partial_warp = threadIdx.x >> 5;
        const unsigned long long output_position = blockIdx.x % noutputs;
        const unsigned long long feature_and_node = blockIdx.x / noutputs;
        const unsigned long long feature_position = feature_and_node % nfeatures;
        const unsigned long long node_chunk = feature_and_node / nfeatures;
        const unsigned long long node_base = node_chunk * nodes_per_chunk;

        extern __shared__ float partial[];
        const unsigned long long partial_size =
            partial_warps * nodes_per_chunk * outputs_per_block * nbins;
        for (unsigned long long index = threadIdx.x;
             index < partial_size;
             index += blockDim.x) {
            partial[index] = 0.0f;
        }
        __syncthreads();

        const unsigned long long feature = col_indexer[feature_position];
        for (unsigned long long base = 0; base < nrows; base += 64) {
            const unsigned long long position =
                base + partial_warp * 32 + lane;
            const bool active = position < nrows;
            const unsigned long long row = active ? row_indexer[position] : 0;
            const int local_node = active ? nodes[row] : -1;
            const unsigned int local_bin =
                active ? arr[row * padded_features + feature] : 0;
            const float local_value = active
                ? target[row * target_outputs + out_indexer[output_position]]
                : 0.0f;

            #pragma unroll
            for (unsigned int source = 0; source < 32; ++source) {
                const int node = __shfl_sync(0xffffffff, local_node, source);
                const unsigned int bin =
                    __shfl_sync(0xffffffff, local_bin, source);
                const float value =
                    __shfl_sync(0xffffffff, local_value, source);
                if (lane == (bin & 31)
                    && node >= (int)node_base
                    && node < (int)(node_base + nodes_per_chunk)
                    && node < (int)nnodes
                    && bin < nbins) {
                    const unsigned int node_offset = node - node_base;
                    const unsigned long long partial_index =
                        ((partial_warp * nodes_per_chunk + node_offset)
                         * nbins + bin);
                    partial[partial_index] += value;
                }
            }
        }
        __syncthreads();

        const unsigned long long combined_size = nodes_per_chunk * nbins;
        for (unsigned long long index = threadIdx.x;
             index < combined_size;
             index += blockDim.x) {
            const unsigned long long bin = index % nbins;
            const unsigned long long node_offset = index / nbins;
            const unsigned long long node = node_base + node_offset;
            if (node < nnodes) {
                const unsigned long long first = node_offset * nbins + bin;
                const unsigned long long second =
                    (nodes_per_chunk + node_offset) * nbins + bin;
                const unsigned long long destination =
                    (((output_position * nnodes + node) * nfeatures
                      + feature_position) * nbins + bin);
                histogram[destination] = partial[first] + partial[second];
            }
        }
    }
    ''',
    'deterministic_histogram',
) if CUDA_FOUND else None

feature_grouper_kernel = cp.ElementwiseKernel(
    """
    uint64 col_indexer
    """,
    'raw bool padded_bool_indexer, raw bool tuple_indexer, raw uint64 padded_col_indexer',
    """
    int tuple_id = col_indexer / 4;
    padded_bool_indexer[col_indexer] = true;
    tuple_indexer[tuple_id] = true;
    padded_col_indexer[col_indexer] = i;

    """,
    'feature_grouper_kernel') if CUDA_FOUND else None


def fill_histogram(res, arr, target, nodes, col_indexer, row_indexer, out_indexer):
    """Fill the histogram res

    Args:
        res: cp.ndarray, histogram of zeros, shape (n_out, n_nodes, n_features, n_bins)
        arr: cp.ndarray, features array, shape (n_data, n_features)
        target: cp.ndarray, values to accumulate, shape (n_data, n_out)
        nodes: cp.ndarray, tree node indices, shape (n_data, )
        col_indexer: cp.ndarray, indices of features to accumulate
        row_indexer: cp.ndarray, indices of rows to accumulate
        out_indexer: cp.ndarray, indices of outputs to accumulate

    Returns:

    """
    # define data split for kernel launch
    nout, nnodes, nfeats, nbins = res.shape

    # padded array of 4 feature tuple
    arr_4t = arr.base.view(dtype=cp.uint32)
    pfeats = arr_4t.shape[1]

    # create 4 feats tuple indexer
    padded_bool_indexer = cp.zeros((arr.base.shape[1],), dtype=cp.bool_)
    padded_col_indexer = cp.zeros((arr.base.shape[1],), dtype=cp.uint64)
    tuple_indexer = cp.zeros((arr_4t.shape[1],), dtype=cp.bool_)

    feature_grouper_kernel(col_indexer, padded_bool_indexer, tuple_indexer, padded_col_indexer)
    tuple_indexer = cp.arange(arr_4t.shape[1], dtype=cp.uint64)[tuple_indexer]

    fb = nfeats * nbins
    nfb = nnodes * fb

    magic_constant = 2 ** 19  # optimal value for my V100

    # split features
    nsplits = math.ceil(nfb / magic_constant)
    # first split by feats
    feats_batch = math.ceil(pfeats / nsplits)
    # split by features
    if feats_batch == nfeats:
        out_batch = magic_constant // nfb
    else:
        out_batch = 1

    ri = row_indexer[:, cp.newaxis, cp.newaxis]
    ti = tuple_indexer[cp.newaxis, :, cp.newaxis]
    oi = out_indexer[cp.newaxis, cp.newaxis, :]

    oii = cp.arange(oi.shape[2], dtype=cp.uint64)[cp.newaxis, cp.newaxis, :]

    for j in range(0, pfeats, feats_batch):
        ti_ = ti[:, j: j + feats_batch]

        for k in range(0, nout, out_batch):
            oi_ = oi[..., k: k + out_batch]
            oii_ = oii[..., k: k + out_batch]

            histogram_kernel_idx(ri, ti_, oi_,
                                 oii_,
                                 padded_col_indexer,
                                 padded_bool_indexer,
                                 target,
                                 arr_4t,
                                 nodes,
                                 nfb, fb, nbins, arr_4t.shape[1], nout,
                                 res)


def _fill_histogram_deterministic(
    res,
    arr,
    target,
    nodes,
    col_indexer,
    row_indexer,
    out_indexer,
    kernel,
    threads,
    outputs_per_block,
    nodes_per_chunk,
    partial_warps,
    profile_record=None,
):
    """Fill a histogram with a fixed row order and no atomic additions."""

    noutputs, nnodes, nfeatures, nbins = res.shape
    if profile_record is None:
        canonical_rows = cp.asarray(
            np.sort(row_indexer.get().astype(np.uint64, copy=False))
        )
    else:
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        started = perf_counter()
        host_rows = row_indexer.get().astype(np.uint64, copy=False)
        profile_record['row_to_host_ms'] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        sorted_rows = np.sort(host_rows)
        profile_record['cpu_sort_ms'] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        canonical_rows = cp.asarray(sorted_rows)
        stream.synchronize()
        profile_record['row_to_device_ms'] = (perf_counter() - started) * 1000.0
    padded_arr = arr.base if isinstance(arr.base, cp.ndarray) else arr
    output_batches = math.ceil(noutputs / outputs_per_block)
    node_chunks = math.ceil(nnodes / nodes_per_chunk)
    blocks = int(output_batches * nfeatures * node_chunks)
    shared_mem = (
        partial_warps
        * nodes_per_chunk
        * outputs_per_block
        * nbins
        * cp.dtype(cp.float32).itemsize
    )
    if profile_record is not None:
        stream.synchronize()
        started = perf_counter()
    kernel(
        (blocks,),
        (threads,),
        (
            padded_arr,
            padded_arr.shape[1],
            target,
            target.shape[1],
            nodes,
            canonical_rows,
            col_indexer.astype(cp.uint64, copy=False),
            out_indexer.astype(cp.uint64, copy=False),
            res,
            canonical_rows.size,
            noutputs,
            nnodes,
            nfeatures,
            nbins,
        ),
        shared_mem=shared_mem,
    )
    if profile_record is not None:
        stream.synchronize()
        profile_record['kernel_ms'] = (perf_counter() - started) * 1000.0
        profile_record['blocks'] = blocks
        profile_record['threads'] = threads
        profile_record['shared_memory_bytes'] = int(shared_mem)


def fill_histogram_deterministic(
    res,
    arr,
    target,
    nodes,
    col_indexer,
    row_indexer,
    out_indexer,
    layout='hybrid',
    profile_record=None,
):
    """Fill a histogram with the best fixed-order layout for its node count."""

    launch = deterministic_histogram_launch(res.shape[1], layout=layout)
    _fill_histogram_deterministic(
        res,
        arr,
        target,
        nodes,
        col_indexer,
        row_indexer,
        out_indexer,
        *launch,
        profile_record=profile_record,
    )


def deterministic_histogram_launch(nnodes, layout='hybrid'):
    """Return one frozen launch geometry under an explicit selection policy."""

    node_count = int(nnodes)
    if layout not in {'perf1', 'hybrid'}:
        raise ValueError(f'unknown deterministic histogram layout: {layout}')
    selected = 'perf1' if layout == 'perf1' or node_count <= 8 else 'perf2'
    if _DETERMINISTIC_HISTOGRAM_LAYOUT_COUNTS is not None:
        key = (str(layout), selected, node_count)
        _DETERMINISTIC_HISTOGRAM_LAYOUT_COUNTS[key] = (
            _DETERMINISTIC_HISTOGRAM_LAYOUT_COUNTS.get(key, 0) + 1
        )
    if selected == 'perf1':
        return deterministic_histogram_perf1_kernel, 128, 2, 8, 2
    return deterministic_histogram_kernel, 64, 1, 16, 2


def fill_histogram_deterministic_perf2_reference(
    res,
    arr,
    target,
    nodes,
    col_indexer,
    row_indexer,
    out_indexer,
    profile_record=None,
):
    """Fill a histogram with the frozen H-HIST-PERF-2 1x16 kernel."""

    _fill_histogram_deterministic(
        res,
        arr,
        target,
        nodes,
        col_indexer,
        row_indexer,
        out_indexer,
        deterministic_histogram_kernel,
        64,
        1,
        16,
        2,
        profile_record,
    )


def fill_histogram_deterministic_perf1_reference(
    res,
    arr,
    target,
    nodes,
    col_indexer,
    row_indexer,
    out_indexer,
    profile_record=None,
):
    """Fill a histogram with the frozen H-HIST-PERF-1 kernel."""

    _fill_histogram_deterministic(
        res,
        arr,
        target,
        nodes,
        col_indexer,
        row_indexer,
        out_indexer,
        deterministic_histogram_perf1_kernel,
        128,
        2,
        8,
        2,
        profile_record,
    )


def fill_histogram_deterministic_reference(
    res,
    arr,
    target,
    nodes,
    col_indexer,
    row_indexer,
    out_indexer,
    profile_record=None,
):
    """Fill a histogram with the frozen H-HIST-DETERMINISM-1 kernel."""

    _fill_histogram_deterministic(
        res,
        arr,
        target,
        nodes,
        col_indexer,
        row_indexer,
        out_indexer,
        deterministic_histogram_reference_kernel,
        64,
        2,
        8,
        2,
        profile_record,
    )


def histogram(arr, gh, nodes, col_indexer, row_indexer, out_indexer, nnodes, max_bins,
              prev_hist=None, small_index=None, big_index=None, deterministic=False,
              deterministic_layout='hybrid', profile_kind=None):
    """Compute grad/hess histogram

    Args:
        arr: cp.ndarray, features array, shape (n_data, n_features)
        gh: cp.ndarray of grad/hess, shape (n_data, n_out + 1), assume the last output is hess, others for grads
        nodes: cp.ndarray, tree node indices, shape (n_data, )
        col_indexer: cp.ndarray, indices of features to accumulate
        row_indexer: cp.ndarray, indices of rows to accumulate
        out_indexer: cp.ndarray, indices of outputs to accumulate
        nnodes: int, total number of nodes in the sample
        max_bins: int, maximum number of feature bins
        prev_hist: cp.ndarray or None, histogram from previous step to apply histogram subtraction trick
        small_index: cp.ndarray or None, node indices to accumulate
        big_index: cp.ndarray or None, node indices to compute by subtraction from prev_hist
        deterministic: If True, accumulate bins in a fixed order without atomics.
        deterministic_layout: ``perf1`` or node-count-based ``hybrid`` layout.

    Returns:
        cp.ndarray, histogram of shape (n_out + 1, n_nodes, n_features, n_bins)
    """

    profile_record = None
    stream = None
    total_started = None
    if deterministic and _DETERMINISTIC_HISTOGRAM_PROFILE is not None:
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        total_started = perf_counter()
        profile_record = {
            'kind': str(profile_kind or 'unspecified'),
            'input_rows': int(row_indexer.shape[0]),
            'outputs': int(out_indexer.shape[0]),
            'nodes': int(nnodes),
            'features': int(col_indexer.shape[0]),
            'bins': int(max_bins),
            'has_previous_histogram': prev_hist is not None,
            'row_filter_ms': 0.0,
            'sibling_subtraction_ms': 0.0,
        }

    nout = out_indexer.shape[0]
    nfeats = col_indexer.shape[0]
    # init union histogram for grad and hess
    # we assume, we have 1 dim hess for one output group

    if profile_record is not None:
        started = perf_counter()
    res = cp.zeros((nout, nnodes, nfeats, max_bins), dtype=cp.float32)
    if profile_record is not None:
        stream.synchronize()
        profile_record['allocation_ms'] = (perf_counter() - started) * 1000.0

    if prev_hist is not None:
        if profile_record is not None:
            started = perf_counter()
        row_indexer = row_indexer[isin(nodes, small_index, row_indexer)]
        if profile_record is not None:
            stream.synchronize()
            profile_record['row_filter_ms'] = (perf_counter() - started) * 1000.0

    if profile_record is not None:
        profile_record['filtered_rows'] = int(row_indexer.shape[0])

    histogram_filler = (
        fill_histogram_deterministic if deterministic else fill_histogram
    )
    if deterministic:
        histogram_filler(
            res,
            arr,
            gh,
            nodes,
            col_indexer,
            row_indexer,
            out_indexer,
            layout=deterministic_layout,
            profile_record=profile_record,
        )
    else:
        histogram_filler(
            res,
            arr,
            gh,
            nodes,
            col_indexer,
            row_indexer,
            out_indexer,
        )
    if profile_record is not None:
        stream.synchronize()
        started = perf_counter()
    res = res.cumsum(axis=-1)
    if profile_record is not None:
        stream.synchronize()
        profile_record['cumsum_ms'] = (perf_counter() - started) * 1000.0

    if prev_hist is not None:
        # indices are paired.
        if profile_record is not None:
            started = perf_counter()
        res[:, big_index] = prev_hist - res[:, big_index + (1 - 2 * (big_index % 2))]
        if profile_record is not None:
            stream.synchronize()
            profile_record['sibling_subtraction_ms'] = (
                perf_counter() - started
            ) * 1000.0

    if profile_record is not None:
        stream.synchronize()
        profile_record['total_ms'] = (perf_counter() - total_started) * 1000.0
        _DETERMINISTIC_HISTOGRAM_PROFILE.append(profile_record)

    return res


def count_histogram(arr, nodes, col_indexer, row_indexer, nnodes, max_bins):
    """Compute exact cumulative row counts for every node, feature, and bin."""

    n_features = int(col_indexer.shape[0])
    counts = cp.zeros(nnodes * n_features * max_bins, dtype=cp.uint64)
    if int(row_indexer.shape[0]) == 0 or n_features == 0:
        return counts.reshape(nnodes, n_features, max_bins)

    row_nodes = nodes[row_indexer].astype(cp.int64, copy=False)[:, cp.newaxis]
    feature_positions = cp.arange(n_features, dtype=cp.int64)[cp.newaxis, :]
    feature_values = arr[row_indexer[:, cp.newaxis], col_indexer[cp.newaxis, :]]
    positions = (
        (row_nodes * n_features + feature_positions) * max_bins + feature_values
    ).astype(cp.int64, copy=False)
    cp.add.at(counts, positions.ravel(), cp.uint64(1))
    return counts.reshape(nnodes, n_features, max_bins).cumsum(axis=-1)


loss_kernel = cp.ElementwiseKernel(
    # inputs
    """
    float32 grad, float32 hess, 
    float32 total_grad, float32 total_hess, 
    float32 nan_grad, float32 nan_hess, 

    uint64 prefix_count, uint64 nan_count, uint64 nodes_count,
    float32 lambda_l2, uint64 min_data_in_leaf
    """,
    # output
    'raw float32 res',

    # kernel
    """
    float rG;
    float G = grad;
    float H = hess;
    unsigned long long C = prefix_count;
    float prev = total_grad * total_grad / (total_hess + lambda_l2);


    if ((C >= min_data_in_leaf) && (nodes_count - C >= min_data_in_leaf)) {
        rG = total_grad - G;
        res[2 * i] = G * G / (H + lambda_l2) + rG * rG / (total_hess - H + lambda_l2) - prev;
    }

    if ((nan_count > 0) && (prefix_count < nodes_count)) {

        C -= nan_count;
        if ((C >= min_data_in_leaf) && (nodes_count - C >= min_data_in_leaf)) {

            G -= nan_grad;
            H -= nan_hess;
            rG = total_grad - G;
            res[2 * i + 1] = G * G / (H + lambda_l2) + rG * rG / (total_hess - H + lambda_l2) - prev;

        }
    }

    """,

    'loss_kernel') if CUDA_FOUND else None


def calc_loss(grad, hess, bin_counts, nodes_count, lambda_l2=0.1, min_data_in_leaf=10):
    """Calculate loss function

    Args:
        grad:
        hess:
        bin_counts: exact cumulative row counts, shape (n_nodes, n_features, n_bins)
        nodes_count:
        lambda_l2:
        min_data_in_leaf:

    Returns:

    """
    # fill loss
    res = cp.zeros(grad.shape + (2,), dtype=cp.float32)
    loss_kernel(grad, hess,
                grad[..., -1:], hess[..., -1:],
                grad[..., :1], hess[..., :1],
                bin_counts[cp.newaxis, ...], bin_counts[cp.newaxis, ..., :1],
                nodes_count[cp.newaxis, :, cp.newaxis, cp.newaxis],
                lambda_l2, int(min_data_in_leaf), res)

    return res


select_among_feature = cp.ElementwiseKernel(
    """
    int64 best_idx, 
    raw float32 loss,
    uint64 binsx2
    """,
    'float32 best_gain, int32 best_split, bool best_nan_left',

    """
    best_gain = loss[i * binsx2 + best_idx];
    best_split = best_idx / 2;
    best_nan_left = (bool) (1 - best_idx % 2);

    """,

    'select_among_feature') if CUDA_FOUND else None

select_total = cp.ElementwiseKernel(
    """
    int64 best_idx,
    raw float32 feat_loss,
    raw int32 feat_split,
    raw bool feat_nan_left,
    raw uint64 col_indexer, 

    uint64 feats
    """,
    'int64 best_feat, float32 best_gain, int32 best_split, bool best_nan_left',

    """
    unsigned long f_ptr = i * feats + best_idx;

    best_gain = feat_loss[f_ptr];
    best_split = feat_split[f_ptr];
    best_nan_left = feat_nan_left[f_ptr];
    best_feat = col_indexer[best_idx];

    """,

    'select_total') if CUDA_FOUND else None


def get_best_split(loss, col_indexer):
    nnodes, nfeats, nbins, _ = loss.shape
    # shape - nnodes * nfeats
    best_loss_among_feat_idx = loss.reshape((nnodes, nfeats, -1)).argmax(axis=-1)

    best_gain, best_split, best_nan_left = \
        select_among_feature(
            best_loss_among_feat_idx, loss, nbins * 2
        )

    best_feat_ = best_gain.argmax(axis=-1)

    best_feat, best_gain, best_split, best_nan_left = \
        select_total(
            best_feat_, best_gain, best_split, best_nan_left, col_indexer, nfeats
        )

    return best_feat, best_gain, best_split, best_nan_left


split_kernel = cp.ElementwiseKernel(
    """
    int32 nodes_index,
    int32 nodes_old,
    raw uint8 arr,
    raw int32 split_nodes,
    raw int64 feat,
    raw int32 split,
    raw bool nan_left,
    uint64 F
    """,
    'int32 nodes',

    """
    if (nodes_index >= 0) {

        int f = feat[nodes_index];
        int s = split[nodes_index];
        bool nl = nan_left[nodes_index];

        int val = arr[i * F + f];

        int d = 0;
        if ((val > s) or ((val == 0) and (not nl))) {d = 1;}

        nodes = split_nodes[2 * nodes_index + d];

    } else {nodes=nodes_old;}

    """,

    'split_kernel') if CUDA_FOUND else None


def make_split(nodes, arr, unique_nodes, split_nodes, feat, split, nan_left, return_pos=False):
    nodes_index = isin(nodes, unique_nodes, return_pos=True)
    new_nodes = split_kernel(nodes_index, nodes, arr, split_nodes, feat, split, nan_left, arr.shape[1])

    if return_pos:
        nodes_index = isin(new_nodes, split_nodes.ravel(), return_pos=True)
        return new_nodes, nodes_index

    return new_nodes


def get_prev_hist(cpu_counts, prev_hist, is_valid_node):
    nbins = prev_hist.shape[-1]
    magic_constant = 20

    is_max = cpu_counts.reshape((-1, 2)).argsort(axis=1).astype(np.bool_).ravel()
    valid = is_max & (cpu_counts > nbins)

    bigger_pos = np.arange(valid.shape[0])[valid]
    smaller_pos = np.arange(valid.shape[0])[~valid]

    if cpu_counts[valid].sum() > (nbins * magic_constant):
        return None, None, None

    bigger_pos = cp.asarray(bigger_pos)
    smaller_pos = cp.asarray(smaller_pos)

    hist_idx = cp.arange(prev_hist.shape[1])[is_valid_node][bigger_pos // 2]

    prev_hist = prev_hist[:, hist_idx]

    return prev_hist, smaller_pos, bigger_pos


apply_values_kernel = cp.ElementwiseKernel(
    """
    uint64 row_indexer,
    uint64 grp_index,

    raw int32 nodes,
    raw T values,
    uint64 ngroups,
    uint64 nout
    """,
    'T pred',

    """
    int out = i % nout;
    int node = nodes[row_indexer * ngroups + grp_index];
    pred = values[node * nout + out];

    """,

    'apply_values_kernel') if CUDA_FOUND else None


def apply_values(nodes, group_index, values):
    row_indexer = cp.arange(nodes.shape[0], dtype=cp.uint64)
    ngroups = nodes.shape[1]
    nout = group_index.shape[0]

    return apply_values_kernel(row_indexer[:, cp.newaxis], group_index[cp.newaxis, :], nodes,
                               values, ngroups, nout)


node_index_kernel = cp.ElementwiseKernel(
    """
    uint64 i_,
    uint64 j_,

    raw T X,

    raw int64 feats,
    raw float32 val_splits,
    raw int32 splits,
    raw bool nan_left,

    uint64 l
    """,
    'int32 node',

    """
    node = 0;

    int f;
    float x;
    int right;
    unsigned long x_ptr;

    while (true) {

        x_ptr = j_ + node;
        f = feats[x_ptr];

        if (f < 0) {return;}

        x = X[i_ * l + f];

        if (isnan(x)) {

            right = 1 - (int) nan_left[x_ptr];

        } else {

            right = (int) (x > val_splits[x_ptr]);

        }

        node = splits[2 * x_ptr + right];

    }

    """,

    'node_index_kernel') if CUDA_FOUND else None

# if F-contiguous array on GPU
generic_tree_prediction_leaves_kernel_f = r'''
    extern "C" __global__
    void tree_prediction_leaves_kernel_f_{TT}(
        const {T}* X,
        const float4* tree,
        const int* gr_subtree_offsets,
        const int n_features,
        const int x_size,
        const int n_gr,
        const int n_buffer,
        int* res)
    {{
        long long th = blockIdx.x * blockDim.x + threadIdx.x;
        long long i_ = th / n_gr;
        if (i_ >= x_size) {{
            return;
        }}
        int j_ = (int)(th % n_gr);

        int tree_offset = gr_subtree_offsets[j_];
        int n_node = 0;
        float4 nd;
        {T} x;
        int n_feat_raw;

        // going through the tree
        while (n_node >= 0) {{
            nd = tree[tree_offset + n_node];

            n_feat_raw = (int)nd.x;
            x = X[x_size * (abs(n_feat_raw) - 1) + i_];

            {comp}
        }}

        // writing result
        res[i_ * n_buffer + j_] = (-n_node - 1);
    }}
    '''

generic_tree_prediction_leaves_kernel = r'''
    extern "C" __global__
    void tree_prediction_leaves_kernel_{TT}(
        const {T}* X,
        const float4* tree,
        const int* gr_subtree_offsets,
        const int n_features,
        const int x_size,
        const int n_gr,
        const int n_buffer, 
        int* res)
    {{
        long long th = blockIdx.x * blockDim.x + threadIdx.x;
        long long i_ = th / n_gr;
        if (i_ >= x_size) {{
            return;
        }}
        int j_ = (int)(th % n_gr);

        long long x_feat_offset = n_features * i_;
        int tree_offset = gr_subtree_offsets[j_];
        int n_node = 0;
        float4 nd;
        {T} x;
        int n_feat_raw;

        // going through the tree
        while (n_node >= 0) {{
            nd = tree[tree_offset + n_node];

            n_feat_raw = (int)nd.x;
            x = X[x_feat_offset + abs(n_feat_raw) - 1];
            
            {comp}
        }}

        // writing result
        res[i_ * n_buffer + j_] = (-n_node - 1);
    }}
    '''

comp_float = '''
                if (isnan(x)) {
                    n_node = (n_feat_raw > 0) ? (int)nd.w : (int)nd.z;
                } else {
                    n_node = ((float)x > nd.y) ? (int)nd.w : (int)nd.z;
                }'''
comp_int = "n_node = ((float)x > nd.y) ? (int)nd.w : (int)nd.z;"
tree_prediction_leaves_typed_kernels = {
    'float32': cp.RawKernel(generic_tree_prediction_leaves_kernel.format(T='float', TT='float', comp=comp_float),
                            'tree_prediction_leaves_kernel_float') if CUDA_FOUND else None,
    'float64': cp.RawKernel(generic_tree_prediction_leaves_kernel.format(T='double', TT='double', comp=comp_float),
                            'tree_prediction_leaves_kernel_double') if CUDA_FOUND else None,
    'int32': cp.RawKernel(generic_tree_prediction_leaves_kernel.format(T='int', TT='int', comp=comp_int),
                          'tree_prediction_leaves_kernel_int') if CUDA_FOUND else None,
    'int64': cp.RawKernel(generic_tree_prediction_leaves_kernel.format(T='long long', TT='longlong', comp=comp_int),
                          'tree_prediction_leaves_kernel_longlong') if CUDA_FOUND else None
}
tree_prediction_leaves_typed_kernels_f = {
    'float32': cp.RawKernel(generic_tree_prediction_leaves_kernel_f.format(T='float', TT='float', comp=comp_float),
                            'tree_prediction_leaves_kernel_f_float') if CUDA_FOUND else None,
    'float64': cp.RawKernel(generic_tree_prediction_leaves_kernel_f.format(T='double', TT='double', comp=comp_float),
                            'tree_prediction_leaves_kernel_f_double') if CUDA_FOUND else None,
    'int32': cp.RawKernel(generic_tree_prediction_leaves_kernel_f.format(T='int', TT='int', comp=comp_int),
                          'tree_prediction_leaves_kernel_f_int') if CUDA_FOUND else None,
    'int64': cp.RawKernel(generic_tree_prediction_leaves_kernel_f.format(T='long long', TT='longlong', comp=comp_int),
                          'tree_prediction_leaves_kernel_f_longlong') if CUDA_FOUND else None
}

tree_prediction_values_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void tree_prediction_values_kernel(
        const int* leaves,
        const unsigned long long* indexes,
        const float* values,
        const int n_out,
        const int x_size,
        const int n_gr,
        float* res)
    {
        long long th = blockIdx.x * blockDim.x + threadIdx.x;
        long long i_ = th / n_out;
        if (i_ >= x_size) {
            return;
        }
        int i_out = (int)(th % n_out);
        
        float prev_val = res[th];
        int i_gr = indexes[i_out];
        int val_offset = leaves[(i_ * n_gr) + i_gr] * n_out;
        float new_val = values[val_offset + i_out];
        res[th] = prev_val + new_val;
    }
    ''',
    'tree_prediction_values_kernel') if CUDA_FOUND else None


def get_tree_node(arr, feats, val_splits, split, nan_left):
    n_gr, nf = feats.shape

    row_indexer = cp.arange(arr.shape[0], dtype=cp.uint64)[:, cp.newaxis]
    out_indexer = cp.arange(0, n_gr * nf, nf, dtype=cp.uint64)[cp.newaxis, :]

    nodes = node_index_kernel(row_indexer, out_indexer, arr, feats,
                              val_splits, split, nan_left, arr.shape[1])

    return nodes


def get_cpu_splitters(unique_nodes, best_feat, best_gain, best_split, best_nan_left, min_gain_to_split=0):
    # print(best_gain.shape, best_gain)

    best_gain = best_gain.get()
    vs = best_gain > min_gain_to_split
    n_vs = vs.sum()

    if n_vs == 0:
        return [], None, None, None, None, None, None

    best_feat = best_feat.get()
    best_split = best_split.get()
    best_nan_left = best_nan_left.get()
    last_node = unique_nodes[-1] + 1

    if n_vs < vs.shape[0]:
        unique_nodes = unique_nodes[vs]
        best_gain = best_gain[vs]
        best_feat = best_feat[vs]
        best_split = best_split[vs]
        best_nan_left = best_nan_left[vs]

    new_nodes_id = np.arange(last_node, last_node + unique_nodes.shape[0] * 2,
                             dtype=np.int32).reshape((-1, 2))

    return unique_nodes, new_nodes_id, best_feat, best_gain, best_split, best_nan_left, vs


def get_gpu_splitters(unique_nodes, new_nodes_id, best_feat, best_split, best_nan_left):
    out = []

    for arr in [unique_nodes, new_nodes_id, best_feat, best_split, best_nan_left]:
        gpu_arr = cp.asarray(arr)
        out.append(gpu_arr)

    new_unique_nodes = new_nodes_id.ravel()

    return out, new_unique_nodes


def _counts_for_rows(node_indexes, row_indexer, nnodes):
    """Count selected rows in active nodes, reserving the last bin as a sentinel."""

    counts = cp.zeros((nnodes + 1,), dtype=np.uint64)
    counts.scatter_add(node_indexes[row_indexer], 1)
    return counts[:-1]


def _active_rows(node_indexes, row_indexer):
    """Remove rows mapped to the inactive-node sentinel (-1)."""

    return row_indexer[node_indexes[row_indexer] >= 0]


def _loss_for_statistics(
    arr,
    grad,
    hess,
    node_indexes,
    row_indexer,
    col_indexer,
    nnodes,
    params,
    return_components=False,
    return_histogram=False,
):
    """Compute split-candidate gains for one explicit row/output statistic set."""

    # ``node_indexes`` uses -1 for rows whose branch stopped at an earlier
    # depth.  The main tree path removes those rows before the next histogram,
    # while diagnostic paths can still pass an all-row indexer.  A negative
    # node id is not a valid histogram bin and must never reach the raw kernel.
    row_indexer = _active_rows(node_indexes, row_indexer)
    n_out = grad.shape[1]
    gh = cp.concatenate((grad, hess), axis=1)
    gh_hist = histogram(
        arr,
        gh,
        node_indexes,
        col_indexer=col_indexer,
        row_indexer=row_indexer,
        out_indexer=cp.arange(gh.shape[1], dtype=cp.uint64),
        nnodes=nnodes,
        max_bins=params['max_bin'],
        deterministic=bool(params.get('deterministic_histogram', False)),
        deterministic_layout=params.get(
            'deterministic_histogram_layout', 'hybrid'
        ),
        profile_kind='diagnostic',
    )
    exact_counts = count_histogram(
        arr,
        node_indexes,
        col_indexer,
        row_indexer,
        nnodes,
        params['max_bin'],
    )
    loss = calc_loss(
        gh_hist[:n_out],
        gh_hist[n_out:],
        exact_counts,
        _counts_for_rows(node_indexes, row_indexer, nnodes),
        lambda_l2=params['lambda_l2'],
        min_data_in_leaf=params['min_data_in_leaf'],
    )
    aggregated = loss.sum(axis=0) if loss.shape[0] > 1 else loss[0]
    if return_components and return_histogram:
        return aggregated, loss, {'gh': gh_hist, 'counts': exact_counts}
    if return_components:
        return aggregated, loss
    if return_histogram:
        return aggregated, {'gh': gh_hist, 'counts': exact_counts}
    return aggregated


def _top_k_split_candidates(loss, col_indexer, top_k):
    """Materialize the top split candidates for every active node."""

    nnodes, nfeats, nbins, nnan_directions = loss.shape
    candidates_per_node = nfeats * nbins * nnan_directions
    k = min(int(top_k), candidates_per_node)
    if k <= 0:
        return [[] for _ in range(nnodes)]
    flattened = loss.reshape((nnodes, candidates_per_node))
    order = cp.argsort(flattened, axis=1)[:, -k:][:, ::-1]
    gains = cp.take_along_axis(flattened, order, axis=1)
    bins_x_directions = nbins * nnan_directions
    feature_positions = order // bins_x_directions
    within_feature = order % bins_x_directions
    split_bins = within_feature // nnan_directions
    nan_left = (within_feature % nnan_directions) == 0
    features = col_indexer[feature_positions]

    features_cpu = features.get()
    gains_cpu = gains.get()
    split_bins_cpu = split_bins.get()
    nan_left_cpu = nan_left.get()
    return [
        [
            {
                'feature': int(features_cpu[node, rank]),
                'split_bin': int(split_bins_cpu[node, rank]),
                'nan_left': bool(nan_left_cpu[node, rank]),
                'gain': float(gains_cpu[node, rank]),
            }
            for rank in range(k)
        ]
        for node in range(nnodes)
    ]


def _top_k_split_candidate_arrays(loss, col_indexer, top_k):
    """Keep the best node-local candidate identities on the GPU."""

    nnodes, nfeats, nbins, nnan_directions = loss.shape
    candidates_per_node = nfeats * nbins * nnan_directions
    k = min(int(top_k), candidates_per_node)
    if k <= 0:
        raise ValueError('top_k must be positive')
    flattened = loss.reshape((nnodes, candidates_per_node))
    partition = cp.argpartition(flattened, candidates_per_node - k, axis=1)[:, -k:]
    partition_gains = cp.take_along_axis(flattened, partition, axis=1)
    descending = cp.argsort(partition_gains, axis=1)[:, ::-1]
    order = cp.take_along_axis(partition, descending, axis=1)
    gains = cp.take_along_axis(flattened, order, axis=1)
    bins_x_directions = nbins * nnan_directions
    feature_positions = order // bins_x_directions
    within_feature = order % bins_x_directions
    return {
        'features': col_indexer[feature_positions].astype(cp.int64, copy=False),
        'split_bins': (within_feature // nnan_directions).astype(cp.int64, copy=False),
        'nan_left': (within_feature % nnan_directions) == 0,
        'diagnostic_gains': gains,
    }


def _hierarchical_top_k_split_candidate_arrays(loss, col_indexer, top_k):
    """Find the exact top-K via feature-local reduction followed by global reduction."""

    nnodes, nfeats, nbins, nnan_directions = loss.shape
    candidates_per_feature = nbins * nnan_directions
    candidates_per_node = nfeats * candidates_per_feature
    k = min(int(top_k), candidates_per_node)
    if k <= 0:
        raise ValueError('top_k must be positive')

    local_k = min(k, candidates_per_feature)
    by_feature = loss.reshape((nnodes, nfeats, candidates_per_feature))
    local_partition = cp.argpartition(
        by_feature,
        candidates_per_feature - local_k,
        axis=2,
    )[:, :, -local_k:]
    local_gains = cp.take_along_axis(by_feature, local_partition, axis=2)
    reduced_gains = local_gains.reshape((nnodes, nfeats * local_k))
    reduced_count = int(reduced_gains.shape[1])
    if reduced_count == k:
        reduced_order = cp.broadcast_to(
            cp.arange(k, dtype=cp.int64)[cp.newaxis, :],
            (nnodes, k),
        )
    else:
        reduced_order = cp.argpartition(
            reduced_gains,
            reduced_count - k,
            axis=1,
        )[:, -k:]
    selected_gains = cp.take_along_axis(reduced_gains, reduced_order, axis=1)
    descending = cp.argsort(selected_gains, axis=1)[:, ::-1]
    reduced_order = cp.take_along_axis(reduced_order, descending, axis=1)
    gains = cp.take_along_axis(reduced_gains, reduced_order, axis=1)

    feature_positions = reduced_order // local_k
    local_positions = reduced_order % local_k
    node_positions = cp.arange(nnodes)[:, cp.newaxis]
    within_feature = local_partition[
        node_positions, feature_positions, local_positions
    ]
    return {
        'features': col_indexer[feature_positions].astype(cp.int64, copy=False),
        'split_bins': (within_feature // nnan_directions).astype(cp.int64, copy=False),
        'nan_left': (within_feature % nnan_directions) == 0,
        'diagnostic_gains': gains,
    }


feature_local_top_k_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void feature_local_top_k(
        const float* loss,
        float* top_values,
        int* top_indices,
        const int candidate_count,
        const int sort_width,
        const int local_k
    ) {
        extern __shared__ unsigned char shared_bytes[];
        float* values = reinterpret_cast<float*>(shared_bytes);
        int* indices = reinterpret_cast<int*>(values + sort_width);
        const int feature_block = blockIdx.x;
        const int input_offset = feature_block * candidate_count;
        const int output_offset = feature_block * local_k;

        for (int position = threadIdx.x; position < sort_width;
             position += blockDim.x) {
            if (position < candidate_count) {
                values[position] = loss[input_offset + position];
                indices[position] = position;
            } else {
                values[position] = -3.402823466e+38F;
                indices[position] = position;
            }
        }
        __syncthreads();

        for (int width = 2; width <= sort_width; width <<= 1) {
            for (int stride = width >> 1; stride > 0; stride >>= 1) {
                for (int position = threadIdx.x; position < sort_width;
                     position += blockDim.x) {
                    const int peer = position ^ stride;
                    if (peer > position) {
                        const bool ascending = (position & width) == 0;
                        const float left_value = values[position];
                        const float right_value = values[peer];
                        const int left_index = indices[position];
                        const int right_index = indices[peer];
                        const bool left_after_right =
                            (left_value > right_value) ||
                            (left_value == right_value && left_index > right_index);
                        const bool left_before_right =
                            (left_value < right_value) ||
                            (left_value == right_value && left_index < right_index);
                        const bool swap = ascending
                            ? left_after_right
                            : left_before_right;
                        if (swap) {
                            values[position] = right_value;
                            values[peer] = left_value;
                            indices[position] = right_index;
                            indices[peer] = left_index;
                        }
                    }
                }
                __syncthreads();
            }
        }

        for (int rank = threadIdx.x; rank < local_k; rank += blockDim.x) {
            const int source = sort_width - 1 - rank;
            top_values[output_offset + rank] = values[source];
            top_indices[output_offset + rank] = indices[source];
        }
    }
    ''',
    'feature_local_top_k',
    options=('--std=c++11',),
)


feature_list_merge_top_k_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void feature_list_merge_top_k(
        const float* local_gains,
        const int* local_indices,
        const long long* columns,
        long long* features,
        long long* split_bins,
        unsigned char* nan_left,
        float* gains,
        const int feature_count,
        const int local_k,
        const int output_k,
        const int nan_directions,
        const int merge_width
    ) {
        extern __shared__ unsigned char shared_bytes[];
        float* head_values = reinterpret_cast<float*>(shared_bytes);
        int* head_features = reinterpret_cast<int*>(head_values + merge_width);
        int* cursors = head_features + merge_width;
        const int node = blockIdx.x;
        const int thread = threadIdx.x;
        const int local_node_offset = node * feature_count * local_k;
        const int output_node_offset = node * output_k;

        if (thread < feature_count) {
            cursors[thread] = 0;
        }
        __syncthreads();

        for (int rank = 0; rank < output_k; ++rank) {
            if (thread < feature_count) {
                const int cursor = cursors[thread];
                const int position = local_node_offset + thread * local_k + cursor;
                head_values[thread] = cursor < local_k
                    ? local_gains[position]
                    : -3.402823466e+38F;
                head_features[thread] = thread;
            } else {
                head_values[thread] = -3.402823466e+38F;
                head_features[thread] = feature_count;
            }
            __syncthreads();

            for (int stride = merge_width >> 1; stride > 0; stride >>= 1) {
                if (thread < stride) {
                    const int peer = thread + stride;
                    const float left_value = head_values[thread];
                    const float right_value = head_values[peer];
                    const int left_feature = head_features[thread];
                    const int right_feature = head_features[peer];
                    if ((right_value > left_value) ||
                        (right_value == left_value && right_feature < left_feature)) {
                        head_values[thread] = right_value;
                        head_features[thread] = right_feature;
                    }
                }
                __syncthreads();
            }

            if (thread == 0) {
                const int feature = head_features[0];
                const int cursor = cursors[feature];
                const int local_position =
                    local_node_offset + feature * local_k + cursor;
                const int within_feature = local_indices[local_position];
                const int output_position = output_node_offset + rank;
                features[output_position] = columns[feature];
                split_bins[output_position] = within_feature / nan_directions;
                nan_left[output_position] =
                    static_cast<unsigned char>((within_feature % nan_directions) == 0);
                gains[output_position] = local_gains[local_position];
                cursors[feature] = cursor + 1;
            }
            __syncthreads();
        }
    }
    ''',
    'feature_list_merge_top_k',
    options=('--std=c++11',),
)


def _fused_hierarchical_top_k_split_candidate_arrays(loss, col_indexer, top_k):
    """Find exact top-K with one shared-memory reduction per feature."""

    nnodes, nfeats, nbins, nnan_directions = loss.shape
    candidates_per_feature = nbins * nnan_directions
    candidates_per_node = nfeats * candidates_per_feature
    k = min(int(top_k), candidates_per_node)
    if k <= 0:
        raise ValueError('top_k must be positive')
    if loss.dtype != cp.float32:
        raise TypeError('fused top-k requires float32 gains')
    sort_width = 1 << (candidates_per_feature - 1).bit_length()
    if sort_width > 1024:
        raise ValueError('fused top-k supports at most 1024 candidates per feature')

    local_k = min(k, candidates_per_feature)
    contiguous_loss = cp.ascontiguousarray(loss)
    local_gains = cp.empty((nnodes, nfeats, local_k), dtype=cp.float32)
    local_indices = cp.empty((nnodes, nfeats, local_k), dtype=cp.int32)
    feature_local_top_k_kernel(
        (nnodes * nfeats,),
        (min(256, sort_width),),
        (
            contiguous_loss,
            local_gains,
            local_indices,
            candidates_per_feature,
            sort_width,
            local_k,
        ),
        shared_mem=sort_width * (cp.dtype(cp.float32).itemsize + cp.dtype(cp.int32).itemsize),
    )

    reduced_gains = local_gains.reshape((nnodes, nfeats * local_k))
    reduced_count = int(reduced_gains.shape[1])
    if reduced_count == k:
        reduced_order = cp.broadcast_to(
            cp.arange(k, dtype=cp.int64)[cp.newaxis, :],
            (nnodes, k),
        )
    else:
        reduced_order = cp.argpartition(
            reduced_gains,
            reduced_count - k,
            axis=1,
        )[:, -k:]
    selected_gains = cp.take_along_axis(reduced_gains, reduced_order, axis=1)
    descending = cp.argsort(selected_gains, axis=1)[:, ::-1]
    reduced_order = cp.take_along_axis(reduced_order, descending, axis=1)
    gains = cp.take_along_axis(reduced_gains, reduced_order, axis=1)

    feature_positions = reduced_order // local_k
    local_positions = reduced_order % local_k
    node_positions = cp.arange(nnodes)[:, cp.newaxis]
    within_feature = local_indices[
        node_positions, feature_positions, local_positions
    ]
    return {
        'features': col_indexer[feature_positions].astype(cp.int64, copy=False),
        'split_bins': (within_feature // nnan_directions).astype(cp.int64, copy=False),
        'nan_left': (within_feature % nnan_directions) == 0,
        'diagnostic_gains': gains,
    }


def _fused_merge_top_k_split_candidate_arrays(loss, col_indexer, top_k):
    """Find exact top-K with feature-local sorting and an exact CUDA merge."""

    nnodes, nfeats, nbins, nnan_directions = loss.shape
    candidates_per_feature = nbins * nnan_directions
    candidates_per_node = nfeats * candidates_per_feature
    k = min(int(top_k), candidates_per_node)
    if k <= 0:
        raise ValueError('top_k must be positive')
    if loss.dtype != cp.float32:
        raise TypeError('fused top-k requires float32 gains')
    sort_width = 1 << (candidates_per_feature - 1).bit_length()
    if sort_width > 1024:
        raise ValueError('fused top-k supports at most 1024 candidates per feature')
    merge_width = 1 << (nfeats - 1).bit_length()
    if merge_width > 1024:
        chunk_results = [
            _fused_merge_top_k_split_candidate_arrays(
                loss[:, start:start + 1024],
                col_indexer[start:start + 1024],
                k,
            )
            for start in range(0, nfeats, 1024)
        ]
        chunk_gains = cp.concatenate(
            [result['diagnostic_gains'] for result in chunk_results],
            axis=1,
        )
        # Each feature chunk is already sorted by descending gain. Keeping K
        # candidates per chunk is sufficient for the exact global top-K. A
        # stable final sort also gives tied candidates a repeatable chunk order.
        global_order = cp.argsort(
            -chunk_gains,
            axis=1,
            kind='stable',
        )[:, :k]
        return {
            name: cp.take_along_axis(
                cp.concatenate([result[name] for result in chunk_results], axis=1),
                global_order,
                axis=1,
            )
            for name in (
                'features',
                'split_bins',
                'nan_left',
                'diagnostic_gains',
            )
        }

    local_k = min(k, candidates_per_feature)
    contiguous_loss = cp.ascontiguousarray(loss)
    contiguous_columns = cp.ascontiguousarray(col_indexer, dtype=cp.int64)
    local_gains = cp.empty((nnodes, nfeats, local_k), dtype=cp.float32)
    local_indices = cp.empty((nnodes, nfeats, local_k), dtype=cp.int32)
    feature_local_top_k_kernel(
        (nnodes * nfeats,),
        (min(256, sort_width),),
        (
            contiguous_loss,
            local_gains,
            local_indices,
            candidates_per_feature,
            sort_width,
            local_k,
        ),
        shared_mem=sort_width * (cp.dtype(cp.float32).itemsize + cp.dtype(cp.int32).itemsize),
    )

    features = cp.empty((nnodes, k), dtype=cp.int64)
    split_bins = cp.empty((nnodes, k), dtype=cp.int64)
    nan_left = cp.empty((nnodes, k), dtype=cp.bool_)
    gains = cp.empty((nnodes, k), dtype=cp.float32)
    feature_list_merge_top_k_kernel(
        (nnodes,),
        (merge_width,),
        (
            local_gains,
            local_indices,
            contiguous_columns,
            features,
            split_bins,
            nan_left,
            gains,
            nfeats,
            local_k,
            k,
            nnan_directions,
            merge_width,
        ),
        shared_mem=(2 * merge_width + nfeats) * cp.dtype(cp.int32).itemsize,
    )
    return {
        'features': features,
        'split_bins': split_bins,
        'nan_left': nan_left,
        'diagnostic_gains': gains,
    }


def _score_candidate_matrix(loss, col_indexer, candidates):
    """Gather oracle gains for a node-by-candidate identity matrix."""

    features = candidates['features']
    split_bins = candidates['split_bins']
    nan_left = candidates['nan_left']
    feature_positions = (
        col_indexer[cp.newaxis, cp.newaxis, :] == features[:, :, cp.newaxis]
    ).argmax(axis=2)
    node_positions = cp.arange(features.shape[0])[:, cp.newaxis]
    direction_positions = cp.where(nan_left, 0, 1)
    return loss[
        node_positions, feature_positions, split_bins, direction_positions
    ]


candidate_statistics_reduction_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void candidate_statistics_reduction(
        const unsigned char* arr,
        const float* grad,
        const float* hess,
        const unsigned long long* sorted_rows,
        const unsigned long long* offsets,
        const unsigned long long* counts,
        const long long* features,
        const long long* split_bins,
        float* prefix,
        float* nan_statistics,
        float* totals,
        unsigned long long* prefix_counts,
        unsigned long long* nan_counts,
        const int nnodes,
        const int shortlist_size,
        const int n_outputs,
        const int n_hess_outputs,
        const unsigned long long arr_row_stride,
        const unsigned long long arr_col_stride,
        const unsigned long long grad_row_stride,
        const unsigned long long grad_col_stride,
        const unsigned long long hess_row_stride,
        const unsigned long long hess_col_stride
    ) {
        const int stat_count = n_outputs + n_hess_outputs;
        int block = blockIdx.x;
        const int stat = block % stat_count;
        block /= stat_count;
        const int candidate = block % shortlist_size;
        const int node = block / shortlist_size;
        if (node >= nnodes) {
            return;
        }

        extern __shared__ unsigned char shared_bytes[];
        float* prefix_shared = reinterpret_cast<float*>(shared_bytes);
        float* nan_shared = prefix_shared + blockDim.x;
        float* total_shared = nan_shared + blockDim.x;
        unsigned long long* prefix_count_shared =
            reinterpret_cast<unsigned long long*>(total_shared + blockDim.x);
        unsigned long long* nan_count_shared = prefix_count_shared + blockDim.x;

        const int thread = threadIdx.x;
        float prefix_sum = 0.0F;
        float nan_sum = 0.0F;
        float total_sum = 0.0F;
        unsigned long long prefix_count = 0;
        unsigned long long nan_count = 0;
        const unsigned long long start = offsets[node];
        const unsigned long long node_count = counts[node];
        const long long feature = features[node * shortlist_size + candidate];
        const long long split_bin = split_bins[node * shortlist_size + candidate];

        for (unsigned long long position = thread; position < node_count;
             position += blockDim.x) {
            const unsigned long long row = sorted_rows[start + position];
            const unsigned char value = arr[
                row * arr_row_stride + feature * arr_col_stride
            ];
            const float statistic = stat < n_outputs
                ? grad[row * grad_row_stride + stat * grad_col_stride]
                : hess[
                    row * hess_row_stride
                    + (stat - n_outputs) * hess_col_stride
                ];
            if (value <= split_bin) {
                prefix_sum += statistic;
                if (stat == 0) {
                    ++prefix_count;
                }
            }
            if (value == 0) {
                nan_sum += statistic;
                if (stat == 0) {
                    ++nan_count;
                }
            }
            if (candidate == 0) {
                total_sum += statistic;
            }
        }

        prefix_shared[thread] = prefix_sum;
        nan_shared[thread] = nan_sum;
        total_shared[thread] = total_sum;
        prefix_count_shared[thread] = prefix_count;
        nan_count_shared[thread] = nan_count;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread < stride) {
                prefix_shared[thread] += prefix_shared[thread + stride];
                nan_shared[thread] += nan_shared[thread + stride];
                total_shared[thread] += total_shared[thread + stride];
                prefix_count_shared[thread] += prefix_count_shared[thread + stride];
                nan_count_shared[thread] += nan_count_shared[thread + stride];
            }
            __syncthreads();
        }

        if (thread == 0) {
            const unsigned long long output =
                (static_cast<unsigned long long>(stat) * nnodes + node)
                * shortlist_size + candidate;
            prefix[output] = prefix_shared[0];
            nan_statistics[output] = nan_shared[0];
            if (candidate == 0) {
                totals[static_cast<unsigned long long>(stat) * nnodes + node] =
                    total_shared[0];
            }
            if (stat == 0) {
                const unsigned long long count_output =
                    static_cast<unsigned long long>(node) * shortlist_size
                    + candidate;
                prefix_counts[count_output] = prefix_count_shared[0];
                nan_counts[count_output] = nan_count_shared[0];
            }
        }
    }
    ''',
    'candidate_statistics_reduction',
    options=('--std=c++11',),
) if CUDA_FOUND else None


def _selected_candidate_kernel_statistics(
    arr,
    grad,
    hess,
    sorted_rows,
    offsets,
    counts,
    features,
    split_bins,
    nnodes,
):
    """Reduce short candidate lists with one segmented CUDA launch."""

    n_outputs = int(grad.shape[1])
    n_hess_outputs = int(hess.shape[1])
    shortlist_size = int(features.shape[1])
    stat_count = n_outputs + n_hess_outputs
    prefix = cp.zeros((stat_count, nnodes, shortlist_size), dtype=cp.float32)
    nan_statistics = cp.zeros_like(prefix)
    totals = cp.zeros((stat_count, nnodes), dtype=cp.float32)
    prefix_counts = cp.zeros((nnodes, shortlist_size), dtype=cp.uint64)
    nan_counts = cp.zeros_like(prefix_counts)
    threads = 128
    blocks = nnodes * shortlist_size * stat_count
    shared_mem = threads * (3 * np.dtype(np.float32).itemsize + 2 * np.dtype(np.uint64).itemsize)
    candidate_statistics_reduction_kernel(
        (blocks,),
        (threads,),
        (
            arr,
            grad,
            hess,
            sorted_rows,
            offsets,
            counts,
            features,
            split_bins,
            prefix,
            nan_statistics,
            totals,
            prefix_counts,
            nan_counts,
            np.int32(nnodes),
            np.int32(shortlist_size),
            np.int32(n_outputs),
            np.int32(n_hess_outputs),
            np.uint64(arr.strides[0] // arr.itemsize),
            np.uint64(arr.strides[1] // arr.itemsize),
            np.uint64(grad.strides[0] // grad.itemsize),
            np.uint64(grad.strides[1] // grad.itemsize),
            np.uint64(hess.strides[0] // hess.itemsize),
            np.uint64(hess.strides[1] // hess.itemsize),
        ),
        shared_mem=shared_mem,
    )
    return prefix, nan_statistics, totals, prefix_counts, nan_counts


def _selected_candidate_float_statistics(
    arr,
    grad,
    hess,
    node_indexes,
    row_indexer,
    features,
    split_bins,
    nnodes,
    counts,
):
    """Reduce all node-local candidate statistics in two batched products."""

    n_outputs = int(grad.shape[1])
    n_hess_outputs = int(hess.shape[1])
    shortlist_size = int(features.shape[1])
    stat_count = n_outputs + n_hess_outputs
    row_nodes = node_indexes[row_indexer]
    if int(row_indexer.size) == 0:
        prefix = cp.zeros((stat_count, nnodes, shortlist_size), dtype=cp.float32)
        return (
            prefix,
            cp.zeros_like(prefix),
            cp.zeros((stat_count, nnodes), dtype=cp.float32),
            cp.zeros((nnodes, shortlist_size), dtype=cp.uint64),
            cp.zeros((nnodes, shortlist_size), dtype=cp.uint64),
        )

    order = cp.argsort(row_nodes)
    sorted_nodes = row_nodes[order].astype(cp.int64, copy=False)
    sorted_rows = row_indexer[order]
    offsets = cp.cumsum(counts, dtype=cp.uint64) - counts
    if shortlist_size <= 64:
        return _selected_candidate_kernel_statistics(
            arr,
            grad,
            hess,
            sorted_rows,
            offsets,
            counts,
            features,
            split_bins,
            nnodes,
        )

    max_rows = int(counts.max().item())
    local_positions = (
        cp.arange(sorted_rows.size, dtype=cp.uint64) - offsets[sorted_nodes]
    ).astype(cp.int64, copy=False)
    grouped_rows = cp.zeros((nnodes, max_rows), dtype=row_indexer.dtype)
    grouped_rows[sorted_nodes, local_positions] = sorted_rows
    valid_rows = (
        cp.arange(max_rows, dtype=cp.uint64)[cp.newaxis, :]
        < counts[:, cp.newaxis]
    )

    feature_values = arr[
        grouped_rows[:, :, cp.newaxis],
        features[:, cp.newaxis, :],
    ]
    valid_candidates = valid_rows[:, :, cp.newaxis]
    prefix_mask = (feature_values <= split_bins[:, cp.newaxis, :]) & valid_candidates
    nan_mask = (feature_values == 0) & valid_candidates
    prefix_counts = prefix_mask.sum(axis=1, dtype=cp.uint64)
    nan_counts = nan_mask.sum(axis=1, dtype=cp.uint64)

    gh_values = cp.concatenate(
        (grad[grouped_rows], hess[grouped_rows]), axis=2
    )
    gh_values *= valid_rows[:, :, cp.newaxis]
    gh_transposed = gh_values.transpose(0, 2, 1)
    prefix = cp.matmul(
        gh_transposed, prefix_mask.astype(cp.float32)
    ).transpose(1, 0, 2)
    nan_statistics = cp.matmul(
        gh_transposed, nan_mask.astype(cp.float32)
    ).transpose(1, 0, 2)
    totals = gh_values.sum(axis=1).T
    return prefix, nan_statistics, totals, prefix_counts, nan_counts


def _selected_candidate_statistics(
    arr,
    grad,
    hess,
    node_indexes,
    row_indexer,
    candidates,
    nnodes,
):
    """Accumulate prefix, NaN, total, and row-count statistics for candidates."""

    features = candidates['features']
    split_bins = candidates['split_bins']
    nan_left = candidates['nan_left']
    if features.ndim != 2 or features.shape[0] != nnodes:
        raise ValueError('candidate identities must have shape (n_nodes, k)')
    if split_bins.shape != features.shape or nan_left.shape != features.shape:
        raise ValueError('candidate identity arrays must have equal shapes')
    shortlist_size = int(features.shape[1])
    if shortlist_size <= 0:
        raise ValueError('candidate shortlist cannot be empty')

    row_indexer = _active_rows(node_indexes, row_indexer)
    counts = _counts_for_rows(node_indexes, row_indexer, nnodes)
    (
        prefix,
        nan_statistics,
        totals,
        prefix_counts,
        nan_counts,
    ) = _selected_candidate_float_statistics(
        arr,
        grad,
        hess,
        node_indexes,
        row_indexer,
        features,
        split_bins,
        nnodes,
        counts,
    )

    return {
        'prefix': prefix,
        'nan': nan_statistics,
        'totals': totals,
        'counts': counts,
        'prefix_counts': prefix_counts,
        'nan_counts': nan_counts,
    }


def _valid_child_counts(total_count, left_count, min_data_in_leaf):
    """Return whether both children contain at least the configured row count."""

    return (
        (left_count >= min_data_in_leaf)
        & ((total_count - left_count) >= min_data_in_leaf)
    )


def _candidate_gain_details(
    statistics,
    candidates,
    n_outputs,
    lambda_l2,
    min_data_in_leaf,
):
    """Transform accumulated candidate statistics into gain components."""

    prefix = statistics['prefix']
    nan_statistics = statistics['nan']
    totals = statistics['totals']
    grad_prefix = prefix[:n_outputs]
    hess_prefix = prefix[n_outputs:]
    nan_grad = nan_statistics[:n_outputs]
    nan_hess = nan_statistics[n_outputs:]
    total_grad = totals[:n_outputs, :, cp.newaxis]
    total_hess = totals[n_outputs:, :, cp.newaxis]
    direction = candidates['nan_left'][cp.newaxis, :, :]
    left_grad = cp.where(direction, grad_prefix, grad_prefix - nan_grad)
    left_hess = cp.where(direction, hess_prefix, hess_prefix - nan_hess)
    counts = statistics['counts']
    prefix_counts = statistics['prefix_counts']
    nan_counts = statistics['nan_counts']
    left_count = cp.where(direction[0], prefix_counts, prefix_counts - nan_counts)
    count_valid = _valid_child_counts(
        counts[:, cp.newaxis], left_count, int(min_data_in_leaf)
    )
    nan_right_valid = (nan_counts > 0) & (prefix_counts < counts[:, cp.newaxis])
    count_valid &= candidates['nan_left'] | nan_right_valid
    valid = cp.broadcast_to(count_valid[cp.newaxis, :, :], left_grad.shape)
    right_grad = total_grad - left_grad
    previous = total_grad * total_grad / (total_hess + lambda_l2)
    component_gains = (
        left_grad * left_grad / (left_hess + lambda_l2)
        + right_grad * right_grad / (total_hess - left_hess + lambda_l2)
        - previous
    )
    component_gains = cp.where(valid, component_gains, cp.float32(0.0))
    return {
        'left_grad': left_grad,
        'left_hess': left_hess,
        'left_count': left_count,
        'valid': valid,
        'component_gains': component_gains,
        'gains': component_gains.sum(axis=0),
    }


def exact_candidate_gains(
    arr,
    grad,
    hess,
    node_indexes,
    row_indexer,
    candidates,
    nnodes,
    lambda_l2,
    min_data_in_leaf,
):
    """Compute full-statistic gains only for explicit node-local candidates."""

    _, details = _exact_candidate_gain_details(
        arr,
        grad,
        hess,
        node_indexes,
        row_indexer,
        candidates,
        nnodes,
        lambda_l2,
        min_data_in_leaf,
    )
    return details['gains']


def _exact_candidate_gain_details(
    arr,
    grad,
    hess,
    node_indexes,
    row_indexer,
    candidates,
    nnodes,
    lambda_l2,
    min_data_in_leaf,
):
    """Return candidate statistics and gain components for diagnostics."""

    statistics = _selected_candidate_statistics(
        arr,
        grad,
        hess,
        node_indexes,
        row_indexer,
        candidates,
        nnodes,
    )
    details = _candidate_gain_details(
        statistics,
        candidates,
        int(grad.shape[1]),
        lambda_l2,
        min_data_in_leaf,
    )
    return statistics, details


def _comparison_summary(observed, reference, rtol=5e-5, atol=5e-3):
    """Summarize one numeric or logical comparison as JSON-compatible data."""

    observed = np.asarray(observed)
    reference = np.asarray(reference)
    if observed.shape != reference.shape:
        return {
            'shape_observed': list(observed.shape),
            'shape_reference': list(reference.shape),
            'allclose': False,
            'mismatch_count': None,
            'maximum_absolute_error': None,
            'maximum_relative_error': None,
        }
    if observed.dtype.kind in 'biu' or reference.dtype.kind in 'biu':
        mismatch = observed != reference
        return {
            'shape': list(observed.shape),
            'allclose': bool(not mismatch.any()),
            'mismatch_count': int(mismatch.sum()),
            'maximum_absolute_error': float(
                np.abs(observed.astype(np.int64) - reference.astype(np.int64)).max(initial=0)
            ),
            'maximum_relative_error': 0.0,
        }
    absolute = np.abs(observed - reference)
    relative = absolute / np.maximum(np.abs(reference), 1e-12)
    return {
        'shape': list(observed.shape),
        'allclose': bool(np.allclose(observed, reference, rtol=rtol, atol=atol)),
        'mismatch_count': int(np.count_nonzero(
            absolute > (atol + rtol * np.abs(reference))
        )),
        'maximum_absolute_error': float(absolute.max(initial=0.0)),
        'maximum_relative_error': float(relative.max(initial=0.0)),
    }


def _cpu_float64_candidate_reference(
    arr,
    grad,
    hess,
    node_indexes,
    row_indexer,
    node_position,
    feature,
    split_bin,
    nan_left,
    lambda_l2,
    min_data_in_leaf,
):
    """Evaluate one candidate with deterministic CPU float64 reductions."""

    node_rows = row_indexer[node_indexes[row_indexer] == node_position]
    feature_values = arr[node_rows, feature].get()
    grad_values = grad[node_rows].get().astype(np.float64, copy=False)
    hess_values = hess[node_rows].get().astype(np.float64, copy=False)
    prefix_mask = feature_values <= split_bin
    nan_mask = feature_values == 0
    prefix_grad = grad_values[prefix_mask].sum(axis=0, dtype=np.float64)
    prefix_hess = hess_values[prefix_mask].sum(axis=0, dtype=np.float64)
    nan_grad = grad_values[nan_mask].sum(axis=0, dtype=np.float64)
    nan_hess = hess_values[nan_mask].sum(axis=0, dtype=np.float64)
    total_grad = grad_values.sum(axis=0, dtype=np.float64)
    total_hess = hess_values.sum(axis=0, dtype=np.float64)
    count = int(node_rows.shape[0])
    prefix_count = int(prefix_mask.sum())
    nan_count = int(nan_mask.sum())

    left_grad = prefix_grad if nan_left else prefix_grad - nan_grad
    left_hess = prefix_hess if nan_left else prefix_hess - nan_hess
    left_count = prefix_count if nan_left else prefix_count - nan_count
    with np.errstate(divide='ignore', invalid='ignore'):
        valid = _valid_child_counts(count, left_count, int(min_data_in_leaf))
        if not nan_left:
            valid &= nan_count > 0 and prefix_count < count
        right_grad = total_grad - left_grad
        previous = total_grad * total_grad / (total_hess + lambda_l2)
        component_gains = (
            left_grad * left_grad / (left_hess + lambda_l2)
            + right_grad * right_grad / (total_hess - left_hess + lambda_l2)
            - previous
        )
    valid = np.full(component_gains.shape, bool(valid), dtype=np.bool_)
    component_gains = np.where(valid, component_gains, 0.0)
    return {
        'prefix': np.concatenate((prefix_grad, prefix_hess)),
        'nan': np.concatenate((nan_grad, nan_hess)),
        'totals': np.concatenate((total_grad, total_hess)),
        'counts': np.asarray(count, dtype=np.int64),
        'prefix_counts': np.asarray(prefix_count, dtype=np.int64),
        'nan_counts': np.asarray(nan_count, dtype=np.int64),
        'left_grad': left_grad,
        'left_hess': left_hess,
        'left_count': np.asarray(left_count, dtype=np.int64),
        'valid': valid,
        'component_gains': component_gains,
        'gain': np.asarray(component_gains.sum(dtype=np.float64)),
    }


def _build_candidate_failure_diagnostics(
    arr,
    full_grad,
    full_hess,
    node_indexes,
    full_rows,
    col_indexer,
    unique_nodes,
    candidates,
    candidate_statistics,
    candidate_details,
    full_histogram,
    refined_indices,
    oracle_indices,
    original_oracle_gains,
    absolute_errors,
    params,
    severe_error_threshold=1.0,
):
    """Locate the first divergent statistic for candidates with a severe error."""

    nnodes = len(unique_nodes)
    n_outputs = int(full_grad.shape[1])
    node_positions = cp.arange(nnodes)[:, cp.newaxis]
    feature_positions = (
        col_indexer[cp.newaxis, cp.newaxis, :] == candidates['features'][:, :, cp.newaxis]
    ).argmax(axis=2)
    full_gh_histogram = full_histogram['gh']
    full_count_histogram = full_histogram['counts']
    reference_statistics = {
        'prefix': full_gh_histogram[
            :, node_positions, feature_positions, candidates['split_bins']
        ],
        'nan': full_gh_histogram[:, node_positions, feature_positions, 0],
        'totals': full_gh_histogram[:, :, 0, -1],
        'counts': _counts_for_rows(
            node_indexes, _active_rows(node_indexes, full_rows), nnodes
        ),
        'prefix_counts': full_count_histogram[
            node_positions, feature_positions, candidates['split_bins']
        ],
        'nan_counts': full_count_histogram[node_positions, feature_positions, 0],
    }
    reference_details = _candidate_gain_details(
        reference_statistics,
        candidates,
        n_outputs,
        params['lambda_l2'],
        params['min_data_in_leaf'],
    )
    candidate_cpu = {
        key: value.get() for key, value in candidate_statistics.items()
    }
    candidate_detail_cpu = {
        key: value.get() for key, value in candidate_details.items()
    }
    reference_cpu = {
        key: value.get() for key, value in reference_statistics.items()
    }
    reference_detail_cpu = {
        key: value.get() for key, value in reference_details.items()
    }
    identity_cpu = {
        key: candidates[key].get()
        for key in ('features', 'split_bins', 'nan_left')
    }
    severe_positions = np.flatnonzero(absolute_errors.max(axis=1) > severe_error_threshold)
    records = []
    stage_order = (
        'counts', 'prefix_counts', 'nan_counts', 'prefix', 'nan', 'totals',
        'left_count', 'left_grad', 'left_hess', 'valid', 'component_gains', 'gain',
    )
    for node_position in severe_positions:
        candidate_indexes = {
            int(absolute_errors[node_position].argmax()),
            int(refined_indices[node_position]),
            int(oracle_indices[node_position]),
        }
        for candidate_index in sorted(candidate_indexes):
            feature = int(identity_cpu['features'][node_position, candidate_index])
            split_bin = int(identity_cpu['split_bins'][node_position, candidate_index])
            nan_left = bool(identity_cpu['nan_left'][node_position, candidate_index])
            float64 = _cpu_float64_candidate_reference(
                arr,
                full_grad,
                full_hess,
                node_indexes,
                full_rows,
                int(node_position),
                feature,
                split_bin,
                nan_left,
                float(params['lambda_l2']),
                float(params['min_data_in_leaf']),
            )
            observed = {
                'prefix': candidate_cpu['prefix'][:, node_position, candidate_index],
                'nan': candidate_cpu['nan'][:, node_position, candidate_index],
                'totals': candidate_cpu['totals'][:, node_position],
                'counts': candidate_cpu['counts'][node_position],
                'prefix_counts': candidate_cpu['prefix_counts'][node_position, candidate_index],
                'nan_counts': candidate_cpu['nan_counts'][node_position, candidate_index],
                'left_count': candidate_detail_cpu['left_count'][node_position, candidate_index],
                'left_grad': candidate_detail_cpu['left_grad'][:, node_position, candidate_index],
                'left_hess': candidate_detail_cpu['left_hess'][:, node_position, candidate_index],
                'valid': candidate_detail_cpu['valid'][:, node_position, candidate_index],
                'component_gains': candidate_detail_cpu['component_gains'][:, node_position, candidate_index],
                'gain': candidate_detail_cpu['gains'][node_position, candidate_index],
            }
            reference = {
                'prefix': reference_cpu['prefix'][:, node_position, candidate_index],
                'nan': reference_cpu['nan'][:, node_position, candidate_index],
                'totals': reference_cpu['totals'][:, node_position],
                'counts': reference_cpu['counts'][node_position],
                'prefix_counts': reference_cpu['prefix_counts'][node_position, candidate_index],
                'nan_counts': reference_cpu['nan_counts'][node_position, candidate_index],
                'left_count': reference_detail_cpu['left_count'][node_position, candidate_index],
                'left_grad': reference_detail_cpu['left_grad'][:, node_position, candidate_index],
                'left_hess': reference_detail_cpu['left_hess'][:, node_position, candidate_index],
                'valid': reference_detail_cpu['valid'][:, node_position, candidate_index],
                'component_gains': reference_detail_cpu['component_gains'][:, node_position, candidate_index],
                'gain': reference_detail_cpu['gains'][node_position, candidate_index],
            }
            comparisons = {
                stage: {
                    'candidate_vs_histogram': _comparison_summary(
                        observed[stage], reference[stage]
                    ),
                    'candidate_vs_float64': _comparison_summary(
                        observed[stage], float64[stage]
                    ),
                    'histogram_vs_float64': _comparison_summary(
                        reference[stage], float64[stage]
                    ),
                }
                for stage in stage_order
            }
            first_divergence = next(
                (
                    stage for stage in stage_order
                    if not comparisons[stage]['candidate_vs_histogram']['allclose']
                ),
                None,
            )
            candidate_gain = float(np.asarray(observed['gain']))
            histogram_gain = float(np.asarray(reference['gain']))
            float64_gain = float(np.asarray(float64['gain']))
            candidate_distance = abs(candidate_gain - float64_gain)
            histogram_distance = abs(histogram_gain - float64_gain)
            closer_reference = (
                'candidate'
                if candidate_distance < histogram_distance
                else 'histogram'
                if histogram_distance < candidate_distance
                else 'equal'
            )
            records.append({
                'node_position': int(node_position),
                'node_id': int(unique_nodes[node_position]),
                'candidate_index': int(candidate_index),
                'identity': [feature, split_bin, nan_left],
                'is_refined_winner': int(candidate_index) == int(refined_indices[node_position]),
                'is_oracle_winner': int(candidate_index) == int(oracle_indices[node_position]),
                'first_divergent_stage': first_divergence,
                'closer_to_float64_gain': closer_reference,
                'candidate_gain': candidate_gain,
                'histogram_gain': histogram_gain,
                'original_oracle_gain': float(
                    original_oracle_gains[node_position, candidate_index]
                ),
                'float64_gain': float64_gain,
                'comparisons': comparisons,
                'values': {
                    stage: {
                        'candidate': np.asarray(observed[stage]).tolist(),
                        'histogram': np.asarray(reference[stage]).tolist(),
                        'float64': np.asarray(float64[stage]).tolist(),
                    }
                    for stage in stage_order
                },
            })
    return {
        'severe_error_threshold': float(severe_error_threshold),
        'severe_node_count': int(len(severe_positions)),
        'records': records,
    }


def _score_splits(loss, col_indexer, features, split_bins, nan_left):
    """Evaluate selected split identities against another gain tensor."""

    feature_positions = (col_indexer[cp.newaxis, :] == features[:, cp.newaxis]).argmax(axis=1)
    node_positions = cp.arange(features.shape[0])
    direction_positions = cp.where(nan_left, 0, 1)
    return loss[node_positions, feature_positions, split_bins, direction_positions]


def _candidate_output_contributions(full_loss_components, col_indexer, candidates):
    """Return full-statistic per-output gains for node-local candidates."""

    if not candidates or not candidates[0]:
        return [], []
    n_nodes = len(candidates)
    k = len(candidates[0])
    if any(len(node_candidates) != k for node_candidates in candidates):
        raise ValueError('candidate lists must have equal lengths')

    features = cp.asarray(
        [[candidate['feature'] for candidate in node] for node in candidates],
        dtype=col_indexer.dtype,
    )
    split_bins = cp.asarray(
        [[candidate['split_bin'] for candidate in node] for node in candidates],
        dtype=cp.int64,
    )
    nan_left = cp.asarray(
        [[candidate['nan_left'] for candidate in node] for node in candidates],
        dtype=cp.bool_,
    )
    feature_positions = (
        col_indexer[cp.newaxis, cp.newaxis, :] == features[:, :, cp.newaxis]
    ).argmax(axis=2)
    node_positions = cp.arange(n_nodes)[:, cp.newaxis]
    direction_positions = cp.where(nan_left, 0, 1)
    values = full_loss_components[
        :, node_positions, feature_positions, split_bins, direction_positions
    ]
    profiles = values.transpose(1, 2, 0)
    full_gains = profiles.sum(axis=2)
    return profiles.get().tolist(), full_gains.get().tolist()


def _leave_one_output_regrets(
    full_loss_components,
    full_loss,
    col_indexer,
    full_best_gain,
    min_gain_to_split,
):
    """Measure full-statistic split regret after removing each output."""

    n_outputs = full_loss_components.shape[0]
    n_nodes = full_loss.shape[0]
    selected_full_gains = cp.empty((n_outputs, n_nodes), dtype=full_loss.dtype)
    for output_index in range(n_outputs):
        reduced_loss = full_loss - full_loss_components[output_index]
        feature, gain, split_bin, nan_left = get_best_split(
            reduced_loss,
            col_indexer,
        )
        candidate_full_gain = _score_splits(
            full_loss,
            col_indexer,
            feature,
            split_bin,
            nan_left,
        )
        selected_full_gains[output_index] = cp.where(
            gain > min_gain_to_split,
            candidate_full_gain,
            0,
        )

    best_cpu = full_best_gain.get()
    selected_cpu = selected_full_gains.get().T
    return [
        [
            (
                normalized_gain_regret(float(best_cpu[node]), float(candidate))
                if float(best_cpu[node]) > min_gain_to_split
                else None
            )
            for candidate in selected_cpu[node]
        ]
        for node in range(n_nodes)
    ]


def _build_split_diagnostic_records(
    diagnostic_loss,
    full_loss,
    col_indexer,
    unique_nodes,
    top_k,
    min_gain_to_split,
    full_loss_components=None,
    selected_output_indices=None,
    include_leave_one_output_regret=False,
):
    """Compare the diagnostic winner with full-statistic candidates per node."""

    selected_feat, selected_gain, selected_split, selected_nan_left = get_best_split(
        diagnostic_loss, col_indexer
    )
    full_feat, full_gain, full_split, full_nan_left = get_best_split(full_loss, col_indexer)
    output_regrets = (
        _leave_one_output_regrets(
            full_loss_components,
            full_loss,
            col_indexer,
            full_gain,
            min_gain_to_split,
        )
        if full_loss_components is not None and include_leave_one_output_regret
        else None
    )
    selected_full_gain = _score_splits(
        full_loss,
        col_indexer,
        selected_feat,
        selected_split,
        selected_nan_left,
    )

    arrays = [
        selected_feat,
        selected_gain,
        selected_split,
        selected_nan_left,
        full_feat,
        full_gain,
        full_split,
        full_nan_left,
        selected_full_gain,
    ]
    (
        selected_feat,
        selected_gain,
        selected_split,
        selected_nan_left,
        full_feat,
        full_gain,
        full_split,
        full_nan_left,
        selected_full_gain,
    ) = [array.get() for array in arrays]
    diagnostic_top_k = _top_k_split_candidates(diagnostic_loss, col_indexer, top_k)
    full_top_k = _top_k_split_candidates(full_loss, col_indexer, top_k)
    output_profiles, candidate_full_gains = (
        _candidate_output_contributions(
            full_loss_components,
            col_indexer,
            diagnostic_top_k,
        )
        if full_loss_components is not None
        else (None, None)
    )
    if candidate_full_gains is not None:
        for node_candidates, node_full_gains in zip(
            diagnostic_top_k,
            candidate_full_gains,
        ):
            for candidate, full_candidate_gain in zip(
                node_candidates,
                node_full_gains,
            ):
                candidate['diagnostic_gain'] = candidate['gain']
                candidate['full_gain'] = float(full_candidate_gain)

    selected_outputs = (
        [int(index) for index in selected_output_indices.get()]
        if selected_output_indices is not None
        else None
    )
    n_outputs = (
        int(full_loss_components.shape[0])
        if full_loss_components is not None
        else None
    )

    records = []
    for position, node_id in enumerate(unique_nodes):
        best_gain = float(full_gain[position])
        candidate_full_gain = float(selected_full_gain[position])
        full_split_available = best_gain > min_gain_to_split
        diagnostic_split_available = (
            float(selected_gain[position]) > min_gain_to_split
        )
        spurious_split = diagnostic_split_available and not full_split_available
        action_full_gain = candidate_full_gain if diagnostic_split_available else 0.0
        regret = action_gain_regret(best_gain, candidate_full_gain,
                                  diagnostic_split_available, full_split_available)
        records.append(
            {
                'node_id': int(node_id),
                'selected_feature': int(selected_feat[position]),
                'selected_split_bin': int(selected_split[position]),
                'selected_nan_left': bool(selected_nan_left[position]),
                'selected_diagnostic_gain': float(selected_gain[position]),
                'selected_full_gain': candidate_full_gain,
                'action_full_gain': action_full_gain,
                'full_best_feature': int(full_feat[position]),
                'full_best_split_bin': int(full_split[position]),
                'full_best_nan_left': bool(full_nan_left[position]),
                'full_best_gain': best_gain,
                'full_split_available': full_split_available,
                'diagnostic_split_available': diagnostic_split_available,
                'spurious_split': spurious_split,
                'same_split': bool(
                    selected_feat[position] == full_feat[position]
                    and selected_split[position] == full_split[position]
                    and selected_nan_left[position] == full_nan_left[position]
                ),
                'regret': regret,
                'diagnostic_top_k': diagnostic_top_k[position],
                'full_top_k': full_top_k[position],
                'candidate_output_contributions': (
                    output_profiles[position] if output_profiles is not None else None
                ),
                'selected_output_indices': selected_outputs,
                'n_outputs': n_outputs,
                'leave_one_output_regret': (
                    output_regrets[position] if output_regrets is not None else None
                ),
            }
        )
    return records


def _timed_gpu_call(function, *args, **kwargs):
    cp.cuda.get_current_stream().synchronize()
    started = perf_counter()
    result = function(*args, **kwargs)
    cp.cuda.get_current_stream().synchronize()
    return result, perf_counter() - started


def _array_fingerprint(*arrays):
    """Return a stable fingerprint for diagnostic CPU or GPU arrays."""

    digest = hashlib.sha256()
    for array in arrays:
        if hasattr(array, 'get'):
            array = array.get()
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode('ascii'))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _top_two_margin(values):
    """Return the non-negative difference between the two largest values."""

    values = np.asarray(values)
    if values.size < 2:
        return None
    top_two = np.partition(values, values.size - 2)[-2:]
    return float(top_two.max() - top_two.min())


def _attach_determinism_frontier(nodes, cpu):
    """Attach ordered K-candidate evidence without changing split selection."""

    for position, node in enumerate(nodes):
        candidates = []
        for rank in range(cpu['features'].shape[1]):
            candidates.append({
                'rank': rank,
                'identity': [
                    int(cpu['features'][position, rank]),
                    int(cpu['split_bins'][position, rank]),
                    bool(cpu['nan_left'][position, rank]),
                ],
                'diagnostic_gain': float(cpu['diagnostic_gains'][position, rank]),
                'refined_gain': float(cpu['refined_gains'][position, rank]),
                'oracle_gain': float(cpu['oracle_candidate_gains'][position, rank]),
            })
        node['candidate_frontier'] = candidates
        node['top1_top2_margins'] = {
            'diagnostic': _top_two_margin(cpu['diagnostic_gains'][position]),
            'refined': _top_two_margin(cpu['refined_gains'][position]),
            'oracle': _top_two_margin(cpu['oracle_candidate_gains'][position]),
        }


def _build_refinement_shadow_entry(
    arr,
    full_grad,
    full_hess,
    node_indexes,
    full_rows,
    col_indexer,
    unique_nodes,
    diagnostic_loss,
    diagnostic_statistics_seconds,
    full_loss,
    full_histogram,
    full_best,
    full_statistics_seconds,
    full_selection_seconds,
    output_budget,
    shortlist_size,
    params,
    capture_failure_diagnostics=False,
    severe_error_threshold=1.0,
    capture_determinism_frontier=False,
):
    """Measure and audit exact shortlist refinement before optional application."""

    candidates, shortlist_seconds = _timed_gpu_call(
        _fused_merge_top_k_split_candidate_arrays,
        diagnostic_loss,
        col_indexer,
        shortlist_size,
    )

    def refine():
        statistics, details = _exact_candidate_gain_details(
            arr,
            full_grad,
            full_hess,
            node_indexes,
            full_rows,
            candidates,
            len(unique_nodes),
            params['lambda_l2'],
            params['min_data_in_leaf'],
        )
        gains = details['gains']
        return gains, gains.argmax(axis=1), statistics, details

    (
        refined_gains,
        refined_indices,
        candidate_statistics,
        candidate_details,
    ), refinement_seconds = _timed_gpu_call(refine)
    oracle_candidate_gains = _score_candidate_matrix(full_loss, col_indexer, candidates)
    oracle_refined_indices = oracle_candidate_gains.argmax(axis=1)
    full_feat, full_gain, full_split, full_nan_left = full_best

    arrays = {
        'features': candidates['features'],
        'split_bins': candidates['split_bins'],
        'nan_left': candidates['nan_left'],
        'diagnostic_gains': candidates['diagnostic_gains'],
        'refined_gains': refined_gains,
        'refined_indices': refined_indices,
        'oracle_candidate_gains': oracle_candidate_gains,
        'oracle_refined_indices': oracle_refined_indices,
        'full_feat': full_feat,
        'full_gain': full_gain,
        'full_split': full_split,
        'full_nan_left': full_nan_left,
    }
    cpu = {name: value.get() for name, value in arrays.items()}
    absolute_errors = np.abs(cpu['refined_gains'] - cpu['oracle_candidate_gains'])
    relative_errors = absolute_errors / np.maximum(
        np.abs(cpu['oracle_candidate_gains']), 1e-6
    )
    candidate_gains_allclose = bool(np.allclose(
        cpu['refined_gains'],
        cpu['oracle_candidate_gains'],
        rtol=5e-5,
        atol=5e-3,
    ))
    nodes = []
    min_gain = float(params['min_gain_to_split'])
    for position, node_id in enumerate(unique_nodes):
        refined_index = int(cpu['refined_indices'][position])
        oracle_refined_index = int(cpu['oracle_refined_indices'][position])
        full_best_gain = float(cpu['full_gain'][position])
        baseline_full_gain = float(cpu['oracle_candidate_gains'][position, 0])
        refined_full_gain = float(cpu['refined_gains'][position, refined_index])
        oracle_refined_gain = float(
            cpu['oracle_candidate_gains'][position, oracle_refined_index]
        )
        baseline_available = float(cpu['diagnostic_gains'][position, 0]) > min_gain
        refined_available = refined_full_gain > min_gain
        full_available = full_best_gain > min_gain
        refined_identity = [
            int(cpu['features'][position, refined_index]),
            int(cpu['split_bins'][position, refined_index]),
            bool(cpu['nan_left'][position, refined_index]),
        ]
        oracle_identity = [
            int(cpu['features'][position, oracle_refined_index]),
            int(cpu['split_bins'][position, oracle_refined_index]),
            bool(cpu['nan_left'][position, oracle_refined_index]),
        ]
        diagnostic_identity = [
            int(cpu['features'][position, 0]),
            int(cpu['split_bins'][position, 0]),
            bool(cpu['nan_left'][position, 0]),
        ]
        tolerance = 5e-3 + 5e-5 * abs(oracle_refined_gain)
        nodes.append({
            'node_id': int(node_id),
            'full_best_gain': full_best_gain,
            'full_split_available': full_available,
            'baseline_regret': action_gain_regret(
                full_best_gain,
                baseline_full_gain,
                baseline_available,
                full_available,
            ),
            'refined_regret': action_gain_regret(
                full_best_gain,
                refined_full_gain,
                refined_available,
                full_available,
            ),
            'refined_full_gain': refined_full_gain,
            'oracle_refined_gain': oracle_refined_gain,
            'refined_identity': refined_identity,
            'diagnostic_identity': diagnostic_identity,
            'would_change_structure_action': bool(
                refined_identity != diagnostic_identity
                or refined_available != baseline_available
            ),
            'applied_to_structure': False,
            'oracle_refined_identity': oracle_identity,
            'refined_winner_gain_equivalent': bool(
                abs(refined_full_gain - oracle_refined_gain) <= tolerance
            ),
            'full_best_identity': [
                int(cpu['full_feat'][position]),
                int(cpu['full_split'][position]),
                bool(cpu['full_nan_left'][position]),
            ],
        })

    if capture_determinism_frontier:
        _attach_determinism_frontier(nodes, cpu)

    two_stage_seconds = (
        diagnostic_statistics_seconds + shortlist_seconds + refinement_seconds
    )
    full_search_seconds = full_statistics_seconds + full_selection_seconds
    result = {
        'shortlist_selector': 'exact_fused_feature_list_merge',
        'diagnostic_output_budget': int(output_budget),
        'shortlist_size': int(shortlist_size),
        'node_count': len(unique_nodes),
        'initial_statistics_seconds': diagnostic_statistics_seconds,
        'shortlist_selection_seconds': shortlist_seconds,
        'refinement_seconds': refinement_seconds,
        'two_stage_seconds': two_stage_seconds,
        'full_statistics_seconds': full_statistics_seconds,
        'full_selection_seconds': full_selection_seconds,
        'full_search_seconds': full_search_seconds,
        'runtime_ratio_to_full_search': two_stage_seconds / full_search_seconds,
        'maximum_candidate_gain_absolute_error': float(absolute_errors.max()),
        'maximum_candidate_gain_relative_error': float(relative_errors.max()),
        'candidate_gains_allclose': candidate_gains_allclose,
        'all_refined_winners_gain_equivalent': all(
            node['refined_winner_gain_equivalent'] for node in nodes
        ),
        'applied_to_structure': False,
        'applied_change_count': 0,
        'application_requested': False,
        'application_eligible': bool(
            candidate_gains_allclose
            and all(node['refined_winner_gain_equivalent'] for node in nodes)
        ),
        'application_refusal_reason': None,
        'nodes': nodes,
    }
    if capture_determinism_frontier:
        result['state_fingerprints'] = {
            'full_grad': _array_fingerprint(full_grad),
            'full_hess': _array_fingerprint(full_hess),
            'node_indexes': _array_fingerprint(node_indexes),
            'full_rows': _array_fingerprint(full_rows),
            'col_indexer': _array_fingerprint(col_indexer),
            'diagnostic_loss': _array_fingerprint(diagnostic_loss),
            'full_loss': _array_fingerprint(full_loss),
            'candidate_identities': _array_fingerprint(
                cpu['features'], cpu['split_bins'], cpu['nan_left']
            ),
        }
    if (
        capture_failure_diagnostics
        and absolute_errors.max(initial=0.0) > severe_error_threshold
    ):
        result['failure_diagnostics'] = _build_candidate_failure_diagnostics(
            arr,
            full_grad,
            full_hess,
            node_indexes,
            full_rows,
            col_indexer,
            unique_nodes,
            candidates,
            candidate_statistics,
            candidate_details,
            full_histogram,
            cpu['refined_indices'],
            cpu['oracle_refined_indices'],
            cpu['oracle_candidate_gains'],
            absolute_errors,
            params,
            severe_error_threshold=severe_error_threshold,
        )
    return result


def _apply_refinement_entry(entry, min_gain_to_split):
    """Convert an audited refinement entry into GPU splitters for this depth."""

    if not entry['candidate_gains_allclose']:
        raise RuntimeError('refined candidate gains failed the registered tolerance')
    if not entry['all_refined_winners_gain_equivalent']:
        raise RuntimeError('refined winners are not gain-equivalent to the oracle')

    nodes = entry['nodes']
    identities = [node['refined_identity'] for node in nodes]
    gains = [node['refined_full_gain'] for node in nodes]
    entry['applied_to_structure'] = True
    entry['applied_change_count'] = int(sum(
        node['would_change_structure_action'] for node in nodes
    ))
    entry['applied_valid_split_count'] = int(sum(
        gain > min_gain_to_split for gain in gains
    ))
    for node in nodes:
        node['applied_to_structure'] = True

    return (
        cp.asarray([identity[0] for identity in identities], dtype=cp.int64),
        cp.asarray(gains, dtype=cp.float32),
        cp.asarray([identity[1] for identity in identities], dtype=cp.int32),
        cp.asarray([identity[2] for identity in identities], dtype=cp.bool_),
    )


def depthwise_grow_tree(
    tree,
    group,
    arr,
    grad,
    hess,
    row_indexer,
    col_indexer,
    params,
    valid_arrs=None,
    split_diagnostics=None,
):
    if valid_arrs is None:
        valid_arrs = []

    # create gh
    n_out = grad.shape[1]
    gh = cp.concatenate((grad, hess), axis=1)
    out_indexer = cp.arange(gh.shape[1], dtype=cp.uint64)

    # init nodes with single zero node
    unique_nodes = np.zeros(1, dtype=np.int32)
    # count unique nodes in active rows
    nodes_count = cp.ones(1, dtype=cp.uint64) * row_indexer.shape[0]
    # nodes for all rows
    nodes = cp.zeros(arr.shape[0], dtype=cp.int32)
    # init valid nodes
    valid_nodes = [cp.zeros(x.shape[0], dtype=cp.int32) for x in valid_arrs]
    # index of node in unique array
    node_indexes = nodes
    prev_hist, small_index, big_index = [None] * 3

    diagnostic_callback = None
    refinement_callback = None
    refinement_shortlists = {}
    apply_refined_splits = False
    capture_determinism_frontier = False
    diagnostic_grad = diagnostic_hess = diagnostic_rows = None
    full_grad = full_hess = None
    diagnostic_top_k = 5
    if split_diagnostics is not None:
        diagnostic_callback = split_diagnostics['callback']
        refinement_callback = split_diagnostics.get('refinement_callback')
        apply_refined_splits = bool(split_diagnostics.get('apply_refined_splits', False))
        capture_determinism_frontier = bool(
            split_diagnostics.get('capture_determinism_frontier', False)
        )
        refinement_failure_diagnostics = bool(
            split_diagnostics.get('refinement_failure_diagnostics', False)
        )
        refinement_severe_error_threshold = float(
            split_diagnostics.get('refinement_severe_error_threshold', 1.0)
        )
        refinement_shortlists = {
            int(budget): int(size)
            for budget, size in split_diagnostics.get('refinement_shortlists', {}).items()
        }
        diagnostic_grad = split_diagnostics['diagnostic_grad']
        diagnostic_hess = split_diagnostics['diagnostic_hess']
        diagnostic_rows = split_diagnostics['diagnostic_rows']
        full_grad = split_diagnostics['full_grad']
        full_hess = split_diagnostics['full_hess']
        diagnostic_top_k = split_diagnostics.get('top_k', 5)
        output_attribution = bool(split_diagnostics.get('output_attribution', False))
        candidate_output_profiles = bool(
            split_diagnostics.get('candidate_output_profiles', False)
        )
        full_rows = cp.arange(arr.shape[0], dtype=cp.uint64)

        def measured_loss(kind, *args, **kwargs):
            cp.cuda.get_current_stream().synchronize()
            started = perf_counter()
            result = _loss_for_statistics(*args, **kwargs)
            cp.cuda.get_current_stream().synchronize()
            timings = split_diagnostics.setdefault('timings', {})
            elapsed = perf_counter() - started
            timings[kind] = timings.get(kind, 0.0) + elapsed
            return result, elapsed

    for niter in range(params['max_depth']):

        nnodes = len(unique_nodes)
        gh_hist = histogram(arr, gh, node_indexes,
                            col_indexer=col_indexer,
                            row_indexer=row_indexer,
                            out_indexer=out_indexer,
                            nnodes=nnodes,
                            max_bins=params['max_bin'],
                            prev_hist=prev_hist,
                            small_index=small_index,
                            big_index=big_index,
                            deterministic=bool(params.get('deterministic_histogram', False)),
                            deterministic_layout=params.get(
                                'deterministic_histogram_layout', 'hybrid'
                            ),
                            profile_kind='structure')
        exact_counts = count_histogram(
            arr,
            node_indexes,
            col_indexer,
            row_indexer,
            nnodes,
            params['max_bin'],
        )

        # assume hess is the last output
        loss = calc_loss(gh_hist[:n_out], gh_hist[n_out:], exact_counts, nodes_count,
                         lambda_l2=params['lambda_l2'],
                         min_data_in_leaf=params['min_data_in_leaf'])

        if loss.shape[0] > 1:
            loss = loss.sum(axis=0)
        else:
            loss = loss[0]

        best_feat, best_gain, best_split, best_nan_left = get_best_split(loss, col_indexer)

        if diagnostic_callback is not None:
            diagnostic_loss, diagnostic_statistics_seconds = measured_loss('current_statistics_seconds',
                arr,
                diagnostic_grad,
                diagnostic_hess,
                node_indexes,
                diagnostic_rows,
                col_indexer,
                nnodes,
                params,
            )
            full_statistics, full_statistics_seconds = measured_loss('full_statistics_seconds',
                arr,
                full_grad,
                full_hess,
                node_indexes,
                full_rows,
                col_indexer,
                nnodes,
                params,
                return_components=(output_attribution or candidate_output_profiles),
                return_histogram=refinement_failure_diagnostics,
            )
            if (output_attribution or candidate_output_profiles) and refinement_failure_diagnostics:
                full_loss, full_loss_components, full_histogram = full_statistics
            elif output_attribution or candidate_output_profiles:
                full_loss, full_loss_components = full_statistics
                full_histogram = None
            elif refinement_failure_diagnostics:
                full_loss, full_histogram = full_statistics
                full_loss_components = None
            else:
                full_loss = full_statistics
                full_loss_components = None
                full_histogram = None
            full_best, full_selection_seconds = _timed_gpu_call(
                get_best_split, full_loss, col_indexer
            )
            records = _build_split_diagnostic_records(
                    diagnostic_loss,
                    full_loss,
                    col_indexer,
                    unique_nodes,
                    diagnostic_top_k,
                    params['min_gain_to_split'],
                    full_loss_components,
                    split_diagnostics.get('selected_output_indices'),
                    include_leave_one_output_regret=output_attribution,
                )
            for record in records:
                record['diagnostic_output_budget'] = split_diagnostics.get('diagnostic_output_budget')
                record['diagnostic_kind'] = 'current'
            diagnostic_callback(group, niter, records)
            diagnostic_budget = int(split_diagnostics.get('diagnostic_output_budget'))
            if refinement_callback is not None and diagnostic_budget in refinement_shortlists:
                entry = _build_refinement_shadow_entry(
                    arr, full_grad, full_hess, node_indexes, full_rows,
                    col_indexer, unique_nodes, diagnostic_loss,
                    diagnostic_statistics_seconds, full_loss, full_histogram, full_best,
                    full_statistics_seconds, full_selection_seconds,
                    diagnostic_budget, refinement_shortlists[diagnostic_budget], params,
                    capture_failure_diagnostics=refinement_failure_diagnostics,
                    severe_error_threshold=refinement_severe_error_threshold,
                    capture_determinism_frontier=capture_determinism_frontier,
                )
                if apply_refined_splits:
                    entry['application_requested'] = True
                    if entry['application_eligible']:
                        best_feat, best_gain, best_split, best_nan_left = (
                            _apply_refinement_entry(entry, params['min_gain_to_split'])
                        )
                    else:
                        entry['application_refusal_reason'] = (
                            'registered_numeric_audit_failed'
                        )
                refinement_callback(group, niter, entry)
            probe = split_diagnostics.get('probe')
            probes = list(split_diagnostics.get('research_probes', ()))
            if probe is not None:
                probes.append(probe)
            for probe in probes:
                if probe['budget'] == split_diagnostics.get('diagnostic_output_budget'):
                    continue
                probe_loss, probe_statistics_seconds = measured_loss('prospective_statistics_seconds',
                    arr, probe['grad'], probe['hess'], node_indexes, probe['rows'],
                    col_indexer, nnodes, params)
                records = _build_split_diagnostic_records(
                    probe_loss, full_loss, col_indexer, unique_nodes, diagnostic_top_k,
                    params['min_gain_to_split'],
                    full_loss_components=(
                        full_loss_components if candidate_output_profiles else None
                    ),
                    selected_output_indices=probe.get('selected_output_indices'),
                    include_leave_one_output_regret=False,
                )
                for record in records:
                    record['diagnostic_output_budget'] = probe['budget']
                    record['diagnostic_kind'] = 'prospective'
                diagnostic_callback(group, niter, records)
                probe_budget = int(probe['budget'])
                if refinement_callback is not None and probe_budget in refinement_shortlists:
                    entry = _build_refinement_shadow_entry(
                        arr, full_grad, full_hess, node_indexes, full_rows,
                        col_indexer, unique_nodes, probe_loss,
                        probe_statistics_seconds, full_loss, full_histogram, full_best,
                        full_statistics_seconds, full_selection_seconds,
                        probe_budget, refinement_shortlists[probe_budget], params,
                        capture_failure_diagnostics=refinement_failure_diagnostics,
                        severe_error_threshold=refinement_severe_error_threshold,
                        capture_determinism_frontier=capture_determinism_frontier,
                    )
                    refinement_callback(group, niter, entry)

        # move to CPU and apply min_gain_to_split condition
        unique_nodes, new_nodes_id, best_feat, best_gain, best_split, best_nan_left, is_valid_node = \
            get_cpu_splitters(unique_nodes, best_feat, best_gain, best_split, best_nan_left,
                              params['min_gain_to_split'])
        # if all nodes are not valid to split - exit
        if len(unique_nodes) == 0:
            break
        # write node info to the Tree
        tree.set_nodes(group, unique_nodes, new_nodes_id, best_feat, best_gain, best_split, best_nan_left)
        # get args back on gpu
        split_args, unique_nodes = get_gpu_splitters(unique_nodes, new_nodes_id,
                                                     best_feat, best_split, best_nan_left)

        # perform split for train set
        nodes, node_indexes = make_split(nodes, arr, *split_args, return_pos=True)
        # perform split for valid sets
        valid_nodes = [make_split(x, y, *split_args, return_pos=False) for (x, y) in zip(valid_nodes, valid_arrs)]

        # update info for the next step
        if niter < (params['max_depth'] - 1):
            # update counts
            nodes_count = cp.zeros((unique_nodes.shape[0] + 1,), dtype=np.uint64)
            nodes_count.scatter_add(node_indexes[row_indexer], 1)
            nodes_count = nodes_count[:-1]
            cpu_counts = nodes_count.get()

            # remove unused rows from indexer
            if cpu_counts.sum() < row_indexer.shape[0]:
                row_indexer = row_indexer[isin(nodes, split_args[1].ravel(), index=row_indexer)]

            # save histogram for the subs trick
            prev_hist, small_index, big_index = get_prev_hist(cpu_counts,
                                                              gh_hist, cp.asarray(is_valid_node))

    return nodes, valid_nodes


accumulate_gh_kernel = cp.ElementwiseKernel(
    """
    uint64 row_indexer,
    uint64 group_index, 

    raw float32 grad, 
    raw float32 hess,
    raw int32 nodes,

    uint64 nout,
    uint64 ngroups,
    uint64 hxst,
    uint64 hyst

    """,
    'raw float32 grad_sum, raw float32 hess_sum',

    """
    int out = i % nout;
    int node = nodes[row_indexer * ngroups + group_index];

    int y_ptr =  node * nout + out;

    atomicAdd(&grad_sum[y_ptr], grad[row_indexer * nout + out]); 
    atomicAdd(&hess_sum[y_ptr], hess[row_indexer * hxst + out * hyst]); 

    """,

    'accumulate_gh_kernel') if CUDA_FOUND else None


deterministic_node_values_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void deterministic_node_values(
        const float* grad,
        const float* hess,
        const int* nodes,
        const unsigned long long* row_indexer,
        const unsigned long long* group_index,
        float* node_values,
        const unsigned long long nrows,
        const unsigned long long nout,
        const unsigned long long ngroups,
        const unsigned long long hxst,
        const unsigned long long hyst,
        const float learning_rate,
        const float lambda_l2)
    {
        const unsigned long long node = blockIdx.x / nout;
        const unsigned long long out = blockIdx.x % nout;
        const unsigned int thread = threadIdx.x;
        float local_grad = 0.0f;
        float local_hess = 0.0f;

        for (unsigned long long position = thread;
             position < nrows;
             position += blockDim.x) {
            const unsigned long long row = row_indexer[position];
            const unsigned long long group = group_index[out];
            if ((unsigned long long)nodes[row * ngroups + group] == node) {
                local_grad += grad[row * nout + out];
                local_hess += hess[row * hxst + out * hyst];
            }
        }

        extern __shared__ float partial[];
        float* partial_grad = partial;
        float* partial_hess = partial + blockDim.x;
        partial_grad[thread] = local_grad;
        partial_hess[thread] = local_hess;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread < stride) {
                partial_grad[thread] += partial_grad[thread + stride];
                partial_hess[thread] += partial_hess[thread + stride];
            }
            __syncthreads();
        }

        if (thread == 0) {
            const unsigned long long output = node * nout + out;
            node_values[output] = -partial_grad[0] * learning_rate
                                  / (partial_hess[0] + lambda_l2);
        }
    }
    ''',
    'deterministic_node_values',
) if CUDA_FOUND else None


def _deterministic_node_values(grad, hess, nodes, row_indexer, group_index,
                               max_nodes, lr, lambda_l2):
    """Calculate leaf values with one stable reduction tree per leaf/output."""

    nout = grad.shape[1]
    ngroups = nodes.shape[1]
    node_values = cp.zeros((max_nodes, nout), dtype=cp.float32)
    # Sorting on the host makes the canonical order explicit and avoids relying
    # on a separate device sort whose implementation may change across CuPy
    # versions. The opt-in diagnostic path transfers only one row-index vector.
    canonical_rows = cp.asarray(
        np.sort(row_indexer.get().astype(np.uint64, copy=False))
    )
    canonical_groups = group_index.astype(cp.uint64, copy=False)
    threads = 256
    blocks = int(max_nodes * nout)
    deterministic_node_values_kernel(
        (blocks,),
        (threads,),
        (
            grad,
            hess,
            nodes,
            canonical_rows,
            canonical_groups,
            node_values,
            canonical_rows.size,
            nout,
            ngroups,
            hess.shape[1],
            int(hess.shape[1] > 1),
            np.float32(lr),
            np.float32(lambda_l2),
        ),
        shared_mem=2 * threads * cp.dtype(cp.float32).itemsize,
    )
    return node_values


def calc_node_values(grad, hess, nodes, row_indexer, group_index, max_nodes,
                     lr=1, lambda_l2=0.1, deterministic=False):
    """Calculate node values based on grad/hess

    Args:
        grad:
        hess:
        nodes:
        row_indexer:
        group_index:
        max_nodes:
        lr:
        lambda_l2:
        deterministic: If True, use a fixed-order reduction without atomics.

    Returns:

    """
    nout = grad.shape[1]
    ngroups = nodes.shape[1]

    if deterministic:
        return _deterministic_node_values(
            grad,
            hess,
            nodes,
            row_indexer,
            group_index,
            max_nodes,
            lr,
            lambda_l2,
        )

    grad_sum = cp.zeros((max_nodes, nout), dtype=cp.float32)
    hess_sum = cp.zeros((max_nodes, nout), dtype=cp.float32)

    accumulate_gh_kernel(row_indexer[:, cp.newaxis], group_index[cp.newaxis, :], grad, hess, nodes,
                         nout, ngroups, hess.shape[1], int(hess.shape[1] > 1),
                         grad_sum, hess_sum)

    node_values = - grad_sum * lr / (hess_sum + lambda_l2)

    return node_values


def pinned_array(array):
    """Move cpu array to the pinned memory

    Args:
        array: np.ndarray

    Returns:
        np.ndarray
    """
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src


@numba.njit
def set_leaf_values(feats, split):
    """Assign leaf indices to the terminal nodes

    Args:
        feats: np.ndarray of tree's split features, shape (n_groups, max_nodes)
        split: shape (ngroups, max_nodes, 2) - node indices corresponding left/right split for the current node

    Returns:
        np.ndarray of leaf indices
    """
    leaves = np.zeros(feats.shape[::-1], dtype=np.int32)
    max_leaves = 0

    for i in range(feats.shape[0]):

        acc = 0

        nodes = [np.int32(0), ]

        while len(nodes) > 0:
            new_nodes = []

            for n in nodes:
                if feats[i, n] == -1:
                    leaves[n, i] = acc
                    acc += 1
                else:
                    new_nodes.extend(split[i, n])

            nodes = new_nodes

        max_leaves = max(max_leaves, acc)

    return leaves, max_leaves
