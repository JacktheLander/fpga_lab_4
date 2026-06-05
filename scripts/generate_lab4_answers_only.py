from pathlib import Path
import csv
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pynq"))

from spmv_utils import csr_spmv, make_sparse_matrix, split_csr_batches, time_call


def read_hls_summary():
    path = ROOT / "reports" / "hls_summary.csv"
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ["target_clock_ns", "estimated_clock_ns"]:
                row[key] = float(row[key])
            for key in ["avg_cycles", "worst_cycles", "bram18", "dsp", "ff", "lut"]:
                row[key] = int(row[key])
            rows[row["top"]] = row
    return rows


def hls_throughput_mops(row, nnz):
    seconds = row["avg_cycles"] * 10e-9
    return (2 * nnz) / seconds / 1e6


def add_page(pdf, title, blocks):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")
    fig.text(0.07, 0.955, title, fontsize=14, fontweight="bold", va="top")
    y = 0.98

    for block in blocks:
        kind = block.get("kind", "para")
        if kind == "prompt":
            y -= 0.006
            lines = textwrap.wrap(block["text"], 92)
            for line in lines:
                ax.text(0, y, line, fontsize=9.3, fontweight="bold", va="top", color="#23395d")
                y -= 0.025
            y -= 0.004
        elif kind == "answer":
            lines = textwrap.wrap(block["text"], 96)
            for line in lines:
                ax.text(0.02, y, line, fontsize=9, va="top")
                y -= 0.023
            y -= 0.006
        elif kind == "bullets":
            for item in block["items"]:
                wrapped = textwrap.wrap(item, 91)
                for idx, line in enumerate(wrapped):
                    prefix = "- " if idx == 0 else "  "
                    ax.text(0.02, y, prefix + line, fontsize=9, va="top")
                    y -= 0.023
            y -= 0.006
        elif kind == "spacer":
            y -= block.get("height", 0.02)

        if y < 0.06:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
            ax.axis("off")
            fig.text(0.07, 0.955, title + " (continued)", fontsize=14, fontweight="bold", va="top")
            y = 0.98

    pdf.savefig(fig)
    plt.close(fig)


def add_table(pdf, title, columns, data, fontsize=8):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.text(0.04, 0.95, title, fontsize=14, fontweight="bold", va="top")
    table = ax.table(cellText=data, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.45)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e7eef8")
        else:
            cell.set_facecolor("#fbfbfb" if row % 2 else "#f2f2f2")
    pdf.savefig(fig)
    plt.close(fig)


def add_bar(pdf, title, labels, values, ylabel, note):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.bar(labels, values, color=["#336699", "#669966", "#996633"])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.06, 0.045, note, fontsize=9)
    fig.tight_layout(rect=[0.04, 0.08, 0.98, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    hls = read_hls_summary()

    csr_main = make_sparse_matrix(size=1000, density=0.001, seed=529)
    x_main = np.arange(1000, dtype=np.int32) % 17 - 8
    sw_csr_main, y_main = time_call(lambda: csr_spmv(csr_main, x_main), repeats=50)
    sw_dense_main, _ = time_call(lambda: csr_main.dense.dot(x_main), repeats=20)

    csr_batch = make_sparse_matrix(size=50, density=0.01, seed=529)
    x_batch = np.arange(50, dtype=np.int32) % 17 - 8
    sw_csr_batch, y_batch = time_call(lambda: csr_spmv(csr_batch, x_batch), repeats=50)
    sw_dense_batch, _ = time_call(lambda: csr_batch.dense.dot(x_batch), repeats=20)

    out = ROOT / "reports" / "ECEN529_Lab4_answers_only.pdf"
    out.parent.mkdir(exist_ok=True)

    with PdfPages(out) as pdf:
        add_page(
            pdf,
            "ECEN 529 Lab 4: Answers Beneath Original Prompts",
            [
                {
                    "kind": "answer",
                    "text": (
                        "This answers-only version follows the original Lab 4 prompt order and omits source-code listings. "
                        "It includes explanations, HLS results, validation status, timing data available from this machine, and the required protocol/batching conclusions."
                    ),
                },
                {
                    "kind": "answer",
                    "text": (
                        "Environment note: Vitis/Vivado 2025.2 is installed, but this local installation does not include the Zynq-7000 "
                        "device part xc7z020clg400-1 used by PYNQ-Z2. Therefore, HLS synthesis and IP export were validated on the installed "
                        "xc7k70tfbg484-1 part. Board-level PYNQ runtime measurements and bitstream generation require a lab install with Zynq support "
                        "and a connected PYNQ board."
                    ),
                },
                {
                    "kind": "bullets",
                    "items": [
                        f"Main sparse test case: 1000 x 1000, density 0.001, NNZ={csr_main.nnz}, output checksum={int(y_main.sum())}.",
                        f"Batch sparse test case: 50 x 50, density 0.01, NNZ={csr_batch.nnz}, output checksum={int(y_batch.sum())}.",
                        "All 14 Vitis HLS C simulations passed: 4 burst kernels, 4 stream kernels, and 6 batch-size variants.",
                    ],
                },
            ],
        )

        add_page(
            pdf,
            "Problem 1(a)",
            [
                {
                    "kind": "prompt",
                    "text": (
                        "Problem 1(a): Implement the sparse matrix-vector multiplication program in Vitis HLS using the AXI-Burst protocol. "
                        "Modify the HLS code to use AXI-Burst and convert the array parameters to pointers."
                    ),
                },
                {
                    "kind": "answer",
                    "text": (
                        "The SPMV kernel was implemented with pointer arguments for rowPtr, columnIndex, values, x, and y. Each pointer is mapped "
                        "to an AXI master interface for DDR access, while num_rows, nnz, size, pointer addresses, and start/done control are mapped "
                        "to AXI-Lite. The kernel first bursts the CSR arrays and x vector into local storage, computes the CSR dot products, and "
                        "bursts the result vector y back to DDR."
                    ),
                },
                {
                    "kind": "answer",
                    "text": (
                        "The HLS reports confirmed inferred AXI master burst reads for rowPtr, columnIndex, values, and x, plus burst writes for y. "
                        "C simulation passed for each burst kernel."
                    ),
                },
                {
                    "kind": "prompt",
                    "text": "Problem 1(a)(ii): Implement four different kernels using each of optimization strategies 1 to 4 in Lecture 4.",
                },
                {
                    "kind": "bullets",
                    "items": [
                        "Strategy 1: pipelined inner-loop multiply-accumulate. This is the simplest version and uses one multiply-add lane.",
                        "Strategy 2: separate multiplication and accumulation phases using a local product buffer. This increases memory use and doubled worst-case latency for this sparse test.",
                        "Strategy 3: partial unrolling with four independent accumulation lanes. It uses more DSPs but needs enough nonzeros per row to realize the benefit.",
                        "Strategy 4: wider unrolled multiply/reduction with partitioned local arrays. It has the most parallel hardware, but for the NNZ≈rows test case its extra fixed overhead is not fully amortized.",
                    ],
                },
            ],
        )

        burst_table = []
        for name in ["spmv_burst_s1", "spmv_burst_s2", "spmv_burst_s3", "spmv_burst_s4"]:
            row = hls[name]
            burst_table.append(
                [
                    name.replace("spmv_burst_", "strategy "),
                    f'{row["estimated_clock_ns"]:.3f}',
                    f'{row["avg_cycles"]:,}',
                    row["worst_time"],
                    row["bram18"],
                    row["dsp"],
                    row["ff"],
                    row["lut"],
                    f"{hls_throughput_mops(row, csr_main.nnz):.2f}",
                ]
            )
        add_table(
            pdf,
            "Problem 1(a) Results: AXI4-Burst HLS Synthesis",
            ["Kernel", "Clk ns", "Avg cycles", "Worst time", "BRAM", "DSP", "FF", "LUT", "Est Mops/s"],
            burst_table,
            fontsize=8,
        )

        add_page(
            pdf,
            "Problem 1(b)",
            [
                {
                    "kind": "prompt",
                    "text": (
                        "Problem 1(b): Write a Python host program to test the four kernels on the PYNQ board. Create Values, ColumnIndex, "
                        "and RowPtr arrays for sparse matrix-vector multiplication."
                    ),
                },
                {
                    "kind": "answer",
                    "text": (
                        "The host flow creates a deterministic sparse matrix, converts it to CSR format, allocates PYNQ buffers for rowPtr, "
                        "columnIndex, values, x, and y, writes the physical buffer addresses to the IP registers, starts each kernel, and reads y "
                        "back for validation."
                    ),
                },
                {
                    "kind": "prompt",
                    "text": "Problem 1(b)(i): Check that the kernel results match the software-only implementation using matrix-vector multiplication.",
                },
                {
                    "kind": "answer",
                    "text": (
                        "The HLS C simulations for all four burst kernels matched the software CSR reference and passed. The generated 1000 x 1000 "
                        f"test matrix has NNZ={csr_main.nnz}; the software output checksum is {int(y_main.sum())}. The PYNQ host program reports a pass/fail "
                        "line for each kernel after comparing every output element."
                    ),
                },
                {
                    "kind": "prompt",
                    "text": (
                        "Problem 1(b)(ii): Measure execution time for each kernel. Throughput is total operations, multiplications and additions, "
                        "per second. Which strategy gives the highest throughput?"
                    ),
                },
                {
                    "kind": "answer",
                    "text": (
                        f"Local software-only timing for the generated 1000 x 1000 case was {sw_csr_main:.6e} s for the Python CSR loop and "
                        f"{sw_dense_main:.6e} s for NumPy dense dot. Board timing could not be collected on this workstation because no PYNQ board "
                        "and no Zynq device support are available locally. Using HLS average-latency estimates at the requested 10 ns clock, "
                        "Strategy 1 has the highest estimated throughput for this very sparse matrix."
                    ),
                },
            ],
        )

        add_page(
            pdf,
            "Problem 1(c)",
            [
                {
                    "kind": "prompt",
                    "text": "Problem 1(c): Modify the HLS code and Python host program to use the AXI-Stream protocol instead of AXI-Burst.",
                },
                {
                    "kind": "answer",
                    "text": (
                        "The stream version uses one AXI4-Stream input and one AXI4-Stream output. The host sends the input stream in this order: "
                        "RowPtr, ColumnIndex, Values, then x. The kernel returns y on the output stream and marks the final result with TLAST. "
                        "An AXI DMA is used between DDR and the streaming kernel."
                    ),
                },
                {
                    "kind": "prompt",
                    "text": (
                        "Problem 1(c)(i): Using the same kernel array size as AXI-Burst, determine execution time with AXI-Stream. Include DMA "
                        "transfer time and sparse matrix setup time. Which protocol provides better performance?"
                    ),
                },
                {
                    "kind": "answer",
                    "text": (
                        "All four stream kernels passed HLS C simulation. Board-level DMA timing could not be measured locally. The HLS estimates "
                        "show that stream kernels have similar compute latency but additional transfer/setup work in the host because the CSR arrays "
                        "and x must be packed into a stream. For this DDR-resident SPMV workload, AXI4-Burst is expected to perform better because the "
                        "kernel can read CSR and x directly from DDR with burst transfers."
                    ),
                },
                {
                    "kind": "prompt",
                    "text": "Problem 1(c)(ii): Which applications are best suited for each protocol type?",
                },
                {
                    "kind": "bullets",
                    "items": [
                        "AXI4-Burst is best for memory-mapped data in DDR, random or indexed access, reusable arrays, and host-managed buffers. SPMV fits this well because x is indexed by columnIndex.",
                        "AXI4-Stream is best for producer-consumer accelerator chains, packetized data, DMA-fed pipelines, sensors, filters, and workloads where each item is consumed in order and intermediate DDR traffic can be avoided.",
                    ],
                },
            ],
        )

        stream_table = []
        for name in ["spmv_stream_s1", "spmv_stream_s2", "spmv_stream_s3", "spmv_stream_s4"]:
            row = hls[name]
            stream_table.append(
                [
                    name.replace("spmv_stream_", "strategy "),
                    f'{row["estimated_clock_ns"]:.3f}',
                    f'{row["avg_cycles"]:,}',
                    row["worst_time"],
                    row["bram18"],
                    row["dsp"],
                    row["ff"],
                    row["lut"],
                    f"{hls_throughput_mops(row, csr_main.nnz):.2f}",
                ]
            )
        add_table(
            pdf,
            "Problem 1(c) Results: AXI4-Stream HLS Synthesis",
            ["Kernel", "Clk ns", "Avg cycles", "Worst time", "BRAM", "DSP", "FF", "LUT", "Est Mops/s"],
            stream_table,
            fontsize=8,
        )

        add_page(
            pdf,
            "Problem 1(d)",
            [
                {
                    "kind": "prompt",
                    "text": (
                        "Problem 1(d): The kernel performs SPMV in batches because local arrays in the kernel are smaller than the matrix and "
                        "vector sizes in the host program."
                    ),
                },
                {
                    "kind": "prompt",
                    "text": "Problem 1(d)(i): Modify the algorithm and/or host program so the algorithm gives correct results with batch processing.",
                },
                {
                    "kind": "answer",
                    "text": (
                        "Batching was corrected by splitting the CSR matrix by complete rows. For each batch, rowPtr is rebased so the first local "
                        "entry is zero, columnIndex remains global because it indexes the full x vector, local values are copied contiguously, and "
                        "the local y result is copied back into the correct global row positions."
                    ),
                },
                {
                    "kind": "prompt",
                    "text": (
                        "Problem 1(d)(ii): Use a 50 x 50 input matrix with around 1 percent non-zero elements. Using three local array sizes, plot "
                        "local kernel array sizes versus execution time of the kernel host program versus software-only Python."
                    ),
                },
                {
                    "kind": "answer",
                    "text": (
                        f"The generated 50 x 50 matrix has NNZ={csr_batch.nnz}. Software-only timing was {sw_csr_batch:.6e} s for the Python CSR loop "
                        f"and {sw_dense_batch:.6e} s for NumPy dense dot. Since PYNQ runtime timing is unavailable locally, the plot uses HLS average "
                        "latency per batch as the hardware kernel-time estimate; the provided PYNQ host program prints measured total times on the board."
                    ),
                },
                {
                    "kind": "prompt",
                    "text": "Problem 1(d)(iii): Identify the HLS configuration that provides the best execution time for the Python host program.",
                },
                {
                    "kind": "answer",
                    "text": (
                        "For this sparse 50 x 50 case, the 16-row / 32-NNZ Strategy 4 burst configuration has the lowest estimated total kernel time. "
                        "Even though it uses four batches, each batch has a very small local latency, and the matrix is sparse enough that larger local "
                        "arrays do not pay off in the HLS estimate."
                    ),
                },
                {
                    "kind": "prompt",
                    "text": "Problem 1(d)(iv): For what matrix sizes does the hardware version outperform or underperform the software-only version?",
                },
                {
                    "kind": "answer",
                    "text": (
                        "Hardware tends to underperform for very small matrices, very low NNZ, or one-off calls because setup, DMA, and register "
                        "programming dominate the useful multiply/add work. Hardware tends to outperform when the matrix is larger, has enough nonzeros "
                        "per row to keep the pipeline busy, or when the same sparse matrix is reused for many x vectors so setup costs are amortized."
                    ),
                },
            ],
        )

        batch_rows = []
        plot_labels = []
        plot_values = []
        for local_rows, max_nnz, burst_name, stream_name in [
            (16, 32, "spmv_burst_s4_rows16", "spmv_stream_s4_rows16"),
            (25, 64, "spmv_burst_s4_rows25", "spmv_stream_s4_rows25"),
            (50, 128, "spmv_burst_s4_rows50", "spmv_stream_s4_rows50"),
        ]:
            batches = split_csr_batches(csr_batch, local_rows, max_nnz)
            for proto, name in [("Burst", burst_name), ("Stream", stream_name)]:
                row = hls[name]
                est_us = len(batches) * row["avg_cycles"] * 10e-3
                batch_rows.append(
                    [
                        proto,
                        local_rows,
                        max_nnz,
                        len(batches),
                        f'{row["avg_cycles"]:,}',
                        f"{est_us:.2f}",
                        row["bram18"],
                        row["dsp"],
                        row["lut"],
                    ]
                )
                if proto == "Burst":
                    plot_labels.append(f"{local_rows} rows")
                    plot_values.append(est_us)

        add_table(
            pdf,
            "Problem 1(d) Results: Batch Array Size Sweep",
            ["Protocol", "Rows", "Max NNZ", "Batches", "Avg cycles/batch", "Est total us", "BRAM", "DSP", "LUT"],
            batch_rows,
            fontsize=8,
        )

        add_bar(
            pdf,
            "Problem 1(d) Plot: Local Burst Array Size vs Estimated Kernel Time",
            plot_labels,
            plot_values,
            "Estimated kernel time (us)",
            "Uses HLS average latency and batch count for the 50 x 50 matrix. Board measurements should replace these estimates when run on PYNQ.",
        )

        add_page(
            pdf,
            "Deliverable Summary",
            [
                {
                    "kind": "prompt",
                    "text": "Deliverables: Submit a single PDF answering all questions in parts (a) to (d).",
                },
                {
                    "kind": "bullets",
                    "items": [
                        "This PDF contains the explanations and results only; source code is intentionally omitted.",
                        "HLS synthesis results are included for burst, stream, and batch kernels.",
                        "Validation result: all Vitis HLS C simulations passed.",
                        "Program-output status: local software outputs were generated and checksummed; PYNQ runtime output requires a connected board and Zynq-enabled Vivado install.",
                        "Vivado design result: block-design recipes are prepared for the burst and stream systems, but final bitstream generation could not be completed on this install because the required PYNQ Zynq-7020 part is missing.",
                    ],
                },
            ],
        )

    print(out)


if __name__ == "__main__":
    main()
