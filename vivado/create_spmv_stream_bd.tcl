# Run from the repository root with:
#   vivado -mode batch -source vivado/create_spmv_stream_bd.tcl
#
# This creates the AXI-Stream design used by pynq/spmv_stream_host.py: Zynq PS,
# one AXI DMA, and one HLS SPMV stream kernel. It requires the Zynq-7000 Vivado
# device files, which were not installed on the workstation used for synthesis.

set script_dir [file dirname [file normalize [info script]]]
set repo_dir [file dirname $script_dir]
set project_dir [file normalize [file join $repo_dir build vivado spmv_stream]]
set part_name [expr {[info exists ::env(PYNQ_PART)] ? $::env(PYNQ_PART) : "xc7z020clg400-1"}]
set kernel_name [expr {[info exists ::env(SPMV_STREAM_KERNEL)] ? $::env(SPMV_STREAM_KERNEL) : "spmv_stream_s4"}]

create_project -force spmv_stream $project_dir -part $part_name
set_property target_language Verilog [current_project]

set ip_dirs [glob -nocomplain [file join $repo_dir build hls $kernel_name solution1 impl ip]]
if {[llength $ip_dirs] == 0} {
    error "No exported HLS IP found for $kernel_name. Run hls/build_hls.ps1 first."
}
set_property ip_repo_paths $ip_dirs [current_project]
update_ip_catalog

create_bd_design spmv_stream_bd
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} \
    [get_bd_cells processing_system7_0]
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_0
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_include_stscntrl_strm {0}] [get_bd_cells axi_dma_0]

set hls_vlnv [lindex [get_ipdefs -all -filter "NAME == $kernel_name"] 0]
create_bd_cell -type ip -vlnv $hls_vlnv ${kernel_name}_0

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins ${kernel_name}_0/ap_clk] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
    [get_bd_pins ${kernel_name}_0/ap_rst_n] \
    [get_bd_pins axi_dma_0/axi_resetn]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins ${kernel_name}_0/in_stream]
connect_bd_intf_net [get_bd_intf_pins ${kernel_name}_0/out_stream] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

foreach slave_pin [list ${kernel_name}_0/s_axi_control axi_dma_0/S_AXI_LITE] {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config {Master "/processing_system7_0/M_AXI_GP0" Clk "/processing_system7_0/FCLK_CLK0"} \
        [get_bd_intf_pins $slave_pin]
}

foreach master_pin {M_AXI_MM2S M_AXI_S2MM} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config {Slave "/processing_system7_0/S_AXI_HP0" Clk "/processing_system7_0/FCLK_CLK0"} \
        [get_bd_intf_pins axi_dma_0/$master_pin]
}

assign_bd_address
validate_bd_design
save_bd_design
make_wrapper -files [get_files [get_property FILE_NAME [current_bd_design]]] -top
add_files -norecurse [glob [file join $project_dir spmv_stream.gen sources_1 bd spmv_stream_bd hdl *wrapper.v]]
update_compile_order -fileset sources_1
write_bd_tcl -force [file join $repo_dir build vivado spmv_stream_bd_export.tcl]
