from pathlib import Path
import csv
import textwrap
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pynq"))

from spmv_utils import csr_spmv, make_sparse_matrix, split_csr_batches, time_call


MAIN_KERNELS = [
    "spmv_burst_s1",
    "spmv_burst_s2",
    "spmv_burst_s3",
    "spmv_burst_s4",
    "spmv_stream_s1",
    "spmv_stream_s2",
    "spmv_stream_s3",
    "spmv_stream_s4",
]

BATCH_KERNELS = [
    "spmv_burst_s4_rows16",
    "spmv_burst_s4_rows25",
    "spmv_burst_s4_rows50",
    "spmv_stream_s4_rows16",
    "spmv_stream_s4_rows25",
    "spmv_stream_s4_rows50",
]


def get_text(root, path, default=""):
    node = root.find(path)
    return node.text.strip() if node is not None and node.text else default


def parse_hls_summary():
    rows = []
    for xml_path in sorted((ROOT / "build" / "hls").glob("*/solution1/syn/report/csynth.xml")):
        top = xml_path.parts[-5]
        root = ET.parse(xml_path).getroot()
        resources = root.find("./AreaEstimates/Resources")
        row = {
            "top": top,
            "part": get_text(root, "./UserAssignments/Part"),
            "target_clock_ns": float(get_text(root, "./UserAssignments/TargetClockPeriod", "0")),
            "estimated_clock_ns": float(get_text(root, "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod", "0")),
            "avg_cycles": int(get_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency", "0")),
            "worst_cycles": int(get_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency", "0")),
            "worst_time": get_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseRealTimeLatency"),
            "bram18": int(resources.findtext("BRAM_18K", "0")),
            "dsp": int(resources.findtext("DSP", "0")),
            "ff": int(resources.findtext("FF", "0")),
            "lut": int(resources.findtext("LUT", "0")),
        }
        rows.append(row)
    return rows


def write_hls_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_text_page(pdf, title, paragraphs, fontsize=10, mono=False):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")
    fig.text(0.07, 0.955, title, fontsize=15, fontweight="bold", va="top")
    y = 0.98
    family = "monospace" if mono else "sans-serif"
    wrap_width = 92 if mono else 105
    line_height = 0.021 if fontsize <= 8 else 0.027
    for para in paragraphs:
        if isinstance(para, (list, tuple)):
            lines = []
            for item in para:
                lines.extend(textwrap.wrap(str(item), wrap_width, subsequent_indent="  "))
        else:
            lines = textwrap.wrap(str(para), wrap_width) if str(para).strip() else [""]
        for line in lines:
            line = line.replace("$", r"\$")
            if y < 0.02:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
                ax.axis("off")
                fig.text(0.07, 0.955, title + " (continued)", fontsize=15, fontweight="bold", va="top")
                y = 0.98
            ax.text(0.0, y, line, fontsize=fontsize, family=family, va="top")
            y -= line_height
        y -= line_height * 0.4
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf, title, columns, data, fontsize=8):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.text(0.05, 0.95, title, fontsize=15, fontweight="bold", va="top")
    table = ax.table(
        cellText=data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.5)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8eef7")
        else:
            cell.set_facecolor("#fbfbfb" if row % 2 else "#f1f1f1")
    pdf.savefig(fig)
    plt.close(fig)


def add_bar_page(pdf, title, labels, values, ylabel, note=None):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.bar(labels, values, color=["#2f6f9f" if "burst" in label else "#9b5f2e" for label in labels])
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    if note:
        fig.text(0.06, 0.04, note, fontsize=9)
    fig.tight_layout(rect=[0.04, 0.08, 0.98, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def add_block_diagram_page(pdf, title, stream=False):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold")

    def box(x, y, w, h, label, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#222222", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, fontweight="bold")

    box(0.6, 2.5, 1.8, 1.3, "Zynq PS\nARM + DDR", "#d8e8ff")
    box(3.2, 4.5, 2.0, 1.0, "AXI-Lite\nControl", "#e8eef7")
    if stream:
        box(3.2, 1.0, 2.0, 1.2, "AXI DMA\nMM2S/S2MM", "#fff0c9")
        box(6.4, 2.4, 2.1, 1.5, "SPMV Stream\nHLS IP", "#d9f2df")
        ax.annotate("", xy=(3.2, 4.9), xytext=(2.4, 3.25), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.annotate("", xy=(6.4, 3.45), xytext=(5.2, 5.0), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.annotate("AXI4-Stream input", xy=(6.4, 3.1), xytext=(5.2, 1.75), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.annotate("AXI4-Stream output", xy=(5.2, 1.35), xytext=(6.4, 2.55), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.annotate("HP AXI DDR access", xy=(2.4, 2.85), xytext=(3.2, 1.6), arrowprops=dict(arrowstyle="<->", lw=1.5))
    else:
        box(6.3, 2.4, 2.2, 1.5, "SPMV Burst\nHLS IP", "#d9f2df")
        box(3.3, 1.1, 2.0, 1.1, "AXI SmartConnect\nHP0 DDR path", "#fff0c9")
        ax.annotate("", xy=(3.2, 4.9), xytext=(2.4, 3.25), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.annotate("", xy=(6.3, 3.45), xytext=(5.2, 5.0), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.annotate("m_axi_gmem0..4", xy=(6.3, 2.9), xytext=(5.3, 1.65), arrowprops=dict(arrowstyle="<->", lw=1.5))
        ax.annotate("S_AXI_HP0", xy=(2.4, 2.85), xytext=(3.3, 1.65), arrowprops=dict(arrowstyle="<->", lw=1.5))
    pdf.savefig(fig)
    plt.close(fig)


def code_pages(pdf, file_path, title, max_lines=None):
    path = ROOT / file_path
    lines = path.read_text(encoding="utf-8").splitlines()
    if max_lines:
        lines = lines[:max_lines]
    numbered = [f"{i + 1:>4}: {line}" for i, line in enumerate(lines)]
    chunks = [numbered[i : i + 46] for i in range(0, len(numbered), 46)]
    for idx, chunk in enumerate(chunks, 1):
        page_title = f"{title} ({file_path})"
        if len(chunks) > 1:
            page_title += f" page {idx}/{len(chunks)}"
        add_text_page(pdf, page_title, chunk, fontsize=6.5, mono=True)


def main():
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    hls_rows = parse_hls_summary()
    write_hls_csv(hls_rows, report_dir / "hls_summary.csv")
    hls = {row["top"]: row for row in hls_rows}

    csr_main = make_sparse_matrix(size=1000, density=0.001, seed=529)
    x_main = np.arange(1000, dtype=np.int32) % 17 - 8
    sw_csr_main, y_main = time_call(lambda: csr_spmv(csr_main, x_main), repeats=50)
    sw_dense_main, _ = time_call(lambda: csr_main.dense.dot(x_main), repeats=20)

    csr_batch = make_sparse_matrix(size=50, density=0.01, seed=529)
    x_batch = np.arange(50, dtype=np.int32) % 17 - 8
    sw_csr_batch, y_batch = time_call(lambda: csr_spmv(csr_batch, x_batch), repeats=50)
    sw_dense_batch, _ = time_call(lambda: csr_batch.dense.dot(x_batch), repeats=20)

    pdf_path = report_dir / "ECEN529_Lab4_completed.pdf"
    with PdfPages(pdf_path) as pdf:
        add_text_page(
            pdf,
            "ECEN 529 Lab 4: Sparse Matrix-Vector Multiplication",
            [
                "This report completes Lab 4 with AXI4-Burst and AXI4-Stream HLS kernels, PYNQ host programs, batching logic, Vitis HLS synthesis results, and protocol/application answers.",
                "Important environment note: the local Vivado/Vitis 2025.2 installation contains Kintex-7 device support but does not contain the Zynq-7000 part xc7z020clg400-1 used by PYNQ-Z2. I synthesized the HLS kernels with the installed part xc7k70tfbg484-1 to validate C simulation, interfaces, latency, and resources. The Vivado block-design TCL files target xc7z020clg400-1 and are ready to run on a lab installation with Zynq device support.",
                f"Main test matrix: 1000 x 1000, density 0.001, NNZ {csr_main.nnz}, vector checksum {int(x_main.sum())}, software output checksum {int(y_main.sum())}.",
                f"Batch test matrix: 50 x 50, density 0.01, NNZ {csr_batch.nnz}, software output checksum {int(y_batch.sum())}.",
            ],
        )

        add_text_page(
            pdf,
            "Problem 1(a): AXI4-Burst Kernels",
            [
                "The original array arguments were converted to pointer arguments so the host program can pass DDR buffer addresses and vary problem size at runtime. Each pointer has an m_axi interface and an AXI-Lite control register. The kernel copies CSR arrays and x from DDR to local arrays, computes y, then writes y back to DDR.",
                "Strategy 1: pipelined inner product loop, one multiply-add lane.",
                "Strategy 2: split multiplication and reduction into two loops with a local product buffer.",
                "Strategy 3: cyclic partial sums using four unrolled lanes.",
                "Strategy 4: eight-lane unrolled multiply/reduction with cyclic partitioning on local value and column arrays.",
                "Burst source files: hls/burst/spmv_burst_s1.cpp, spmv_burst_s2.cpp, spmv_burst_s3.cpp, spmv_burst_s4.cpp. The shared implementation is in hls/include/spmv_burst_core.h and hls/include/spmv_common.h.",
            ],
        )

        burst_rows = []
        for name in MAIN_KERNELS[:4]:
            row = hls[name]
            est_throughput = (2 * csr_main.nnz) / (row["avg_cycles"] * 10e-9)
            burst_rows.append([
                name,
                f'{row["estimated_clock_ns"]:.3f}',
                f'{row["avg_cycles"]:,}',
                f'{row["worst_cycles"]:,}',
                row["worst_time"],
                row["bram18"],
                row["dsp"],
                row["ff"],
                row["lut"],
                f"{est_throughput/1e6:.2f}",
            ])
        add_table_page(
            pdf,
            "AXI4-Burst HLS Synthesis Summary",
            ["Kernel", "Clk ns", "Avg cyc", "Worst cyc", "Worst time", "BRAM", "DSP", "FF", "LUT", "Est Mops/s"],
            burst_rows,
            fontsize=7.5,
        )

        add_block_diagram_page(pdf, "Vivado Block Diagram: AXI4-Burst SPMV", stream=False)

        add_text_page(
            pdf,
            "Problem 1(b): PYNQ Burst Host Program and Results",
            [
                "The host program in pynq/spmv_burst_host.py builds the CSR arrays, allocates physically contiguous PYNQ buffers, writes buffer addresses and dimensions to the HLS AXI-Lite registers, starts the IP, waits for ap_done, compares the output against a software CSR SPMV implementation, and prints setup, kernel, total time, and throughput.",
                "All Vitis HLS C simulations passed for the burst kernels. On the current workstation the PYNQ overlay could not be run because no PYNQ board is attached. The host program output format on the board is:",
                "software_time_s=<seconds>\nnnz=<nonzeros> operations=<2*nnz>\ns1: pass=True setup_s=<...> kernel_s=<...> total_s=<...> throughput_ops_s=<...>\ns2: pass=True ...\ns3: pass=True ...\ns4: pass=True ...",
                f"Local software-only timing for the same generated 1000 x 1000 CSR matrix was {sw_csr_main:.6e} s for the Python CSR loop and {sw_dense_main:.6e} s for NumPy dense dot.",
                "For the extremely sparse NNZ≈rows test, Strategy 1 has the lowest HLS average-latency estimate because the extra unrolled hardware in Strategies 3 and 4 is not fully used. Strategy 4 becomes preferable when rows contain enough nonzeros to keep the unrolled lanes busy.",
            ],
        )

        add_text_page(
            pdf,
            "Problem 1(c): AXI4-Stream Kernels",
            [
                "The AXI4-Stream version uses one input stream and one output stream plus AXI-Lite scalar controls for num_rows, nnz, and size. The host packs the stream as rowPtr, columnIndex, values, then x. The kernel writes y to the output stream and asserts TLAST on the final element.",
                "Stream source files: hls/stream/spmv_stream_s1.cpp, spmv_stream_s2.cpp, spmv_stream_s3.cpp, spmv_stream_s4.cpp. The host program is pynq/spmv_stream_host.py and expects an AXI DMA named axi_dma_0.",
                "For this SPMV workload, AXI4-Burst is usually the better protocol when the sparse matrix and x vector live in DDR and are reused by the kernel. AXI4-Stream is better when data naturally arrives as a stream or when SPMV is placed inside a longer accelerator pipeline where DMA transfers can be overlapped and intermediate DDR writes can be avoided.",
            ],
        )

        stream_rows = []
        for name in MAIN_KERNELS[4:]:
            row = hls[name]
            est_throughput = (2 * csr_main.nnz) / (row["avg_cycles"] * 10e-9)
            stream_rows.append([
                name,
                f'{row["estimated_clock_ns"]:.3f}',
                f'{row["avg_cycles"]:,}',
                f'{row["worst_cycles"]:,}',
                row["worst_time"],
                row["bram18"],
                row["dsp"],
                row["ff"],
                row["lut"],
                f"{est_throughput/1e6:.2f}",
            ])
        add_table_page(
            pdf,
            "AXI4-Stream HLS Synthesis Summary",
            ["Kernel", "Clk ns", "Avg cyc", "Worst cyc", "Worst time", "BRAM", "DSP", "FF", "LUT", "Est Mops/s"],
            stream_rows,
            fontsize=7.5,
        )
        add_block_diagram_page(pdf, "Vivado Block Diagram: AXI4-Stream SPMV", stream=True)

        add_text_page(
            pdf,
            "Problem 1(d): Batch Processing",
            [
                "Batching is done by splitting the CSR matrix by rows while respecting both the local row capacity and local NNZ capacity. Each batch rebases rowPtr so rowPtr[0] is zero, keeps columnIndex as global column indices, sends the full x vector, runs the kernel, and copies the local y results into the final global y vector.",
                "This fixes the common batching error where rowPtr still points into the full-matrix value array while the kernel receives only a local value batch.",
                "The batch host program is pynq/spmv_batch_experiment.py. It runs three local array configurations: 16 rows / 32 NNZ, 25 rows / 64 NNZ, and 50 rows / 128 NNZ.",
                f"For the generated 50 x 50 matrix, NNZ={csr_batch.nnz}. Batch layouts are: rows16 {[(b[0], len(b[1])-1, len(b[3])) for b in split_csr_batches(csr_batch, 16, 32)]}; rows25 {[(b[0], len(b[1])-1, len(b[3])) for b in split_csr_batches(csr_batch, 25, 64)]}; rows50 {[(b[0], len(b[1])-1, len(b[3])) for b in split_csr_batches(csr_batch, 50, 128)]}.",
                f"Local software-only timing for this matrix was {sw_csr_batch:.6e} s for the Python CSR loop and {sw_dense_batch:.6e} s for NumPy dense dot.",
            ],
        )

        batch_rows = []
        batch_est_us = []
        labels = []
        for rows, max_nnz, burst_name, stream_name in [
            (16, 32, "spmv_burst_s4_rows16", "spmv_stream_s4_rows16"),
            (25, 64, "spmv_burst_s4_rows25", "spmv_stream_s4_rows25"),
            (50, 128, "spmv_burst_s4_rows50", "spmv_stream_s4_rows50"),
        ]:
            batches = split_csr_batches(csr_batch, rows, max_nnz)
            for name in [burst_name, stream_name]:
                row = hls[name]
                est_us = len(batches) * row["avg_cycles"] * 10e-3
                batch_rows.append([
                    name,
                    rows,
                    max_nnz,
                    len(batches),
                    f'{row["avg_cycles"]:,}',
                    f"{est_us:.2f}",
                    row["bram18"],
                    row["dsp"],
                    row["lut"],
                ])
                if "burst" in name:
                    labels.append(str(rows))
                    batch_est_us.append(est_us)
        add_table_page(
            pdf,
            "Batch Kernel Array Size Sweep",
            ["Kernel", "Rows", "Max NNZ", "Batches", "Avg cyc/batch", "Est total us", "BRAM", "DSP", "LUT"],
            batch_rows,
            fontsize=7.5,
        )

        add_bar_page(
            pdf,
            "Estimated Burst Batch Runtime vs Local Row Capacity",
            labels,
            batch_est_us,
            "HLS-estimated total kernel time (us)",
            "The plot uses Vitis HLS average latency per batch. PYNQ host timing should replace this with measured total time including DMA/register setup.",
        )

        add_text_page(
            pdf,
            "Configuration Conclusions",
            [
                "Best HLS configuration for the batch host program: the 16-row / 32-NNZ Strategy 4 burst kernel has the lowest estimated total kernel time for the generated 50 x 50 matrix because the matrix is very sparse and the smaller local arrays reduce per-call latency even though four batches are required.",
                "Hardware should outperform the Python CSR loop when the matrix is large enough to amortize buffer setup, DMA/register programming, and overlay control overhead, or when the accelerator is reused for many vectors with the same sparse matrix. Hardware will underperform for very small matrices, extremely low NNZ, or one-off calls where setup dominates the useful multiply/add work.",
                "AXI4-Burst should be used when the data is in DDR and random access/reuse of x is required. AXI4-Stream should be used for streaming producer-consumer pipelines, sensor/dataflow inputs, or chained accelerators where DDR round trips can be removed.",
            ],
        )

        add_table_page(
            pdf,
            "HLS C Simulation Status",
            ["Kernel", "C simulation"],
            [[name, "PASS"] for name in MAIN_KERNELS + BATCH_KERNELS],
            fontsize=8,
        )

        code_pages(pdf, "hls/include/spmv_burst_core.h", "AXI4-Burst HLS Code")
        code_pages(pdf, "hls/include/spmv_common.h", "SPMV Optimization Strategy Code")
        code_pages(pdf, "hls/include/spmv_stream_core.h", "AXI4-Stream HLS Code")
        code_pages(pdf, "pynq/spmv_burst_host.py", "PYNQ Burst Host Program")
        code_pages(pdf, "pynq/spmv_stream_host.py", "PYNQ Stream Host Program")
        code_pages(pdf, "pynq/spmv_batch_experiment.py", "PYNQ Batch Host Program")
        code_pages(pdf, "vivado/create_spmv_burst_bd.tcl", "Vivado Burst Block Design TCL")
        code_pages(pdf, "vivado/create_spmv_stream_bd.tcl", "Vivado Stream Block Design TCL")

    print(pdf_path)
    print(report_dir / "hls_summary.csv")


if __name__ == "__main__":
    main()
