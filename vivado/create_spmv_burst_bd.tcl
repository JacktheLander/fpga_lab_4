# Run from the repository root with:
#   vivado -mode batch -source vivado/create_spmv_burst_bd.tcl
#
# This script targets PYNQ-Z2/PYNQ-Z1 style Zynq-7020 hardware. It requires the
# Zynq-7000 device files to be installed in Vivado. The workstation used to
# prepare this solution only had Kintex-7 devices installed, so this script is
# provided as the reproducible Vivado block-design recipe for the lab machine.

set script_dir [file dirname [file normalize [info script]]]
set repo_dir [file dirname $script_dir]
set project_dir [file normalize [file join $repo_dir build vivado spmv_burst]]
set part_name [expr {[info exists ::env(PYNQ_PART)] ? $::env(PYNQ_PART) : "xc7z020clg400-1"}]
set kernel_name [expr {[info exists ::env(SPMV_BURST_KERNEL)] ? $::env(SPMV_BURST_KERNEL) : "spmv_burst_s4"}]

create_project -force spmv_burst $project_dir -part $part_name
set_property target_language Verilog [current_project]

set ip_dirs [glob -nocomplain [file join $repo_dir build hls $kernel_name solution1 impl ip]]
if {[llength $ip_dirs] == 0} {
    error "No exported HLS IP found for $kernel_name. Run hls/build_hls.ps1 first."
}
set_property ip_repo_paths $ip_dirs [current_project]
update_ip_catalog

create_bd_design spmv_burst_bd
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} \
    [get_bd_cells processing_system7_0]
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]

set hls_vlnv [lindex [get_ipdefs -all -filter "NAME == $kernel_name"] 0]
create_bd_cell -type ip -vlnv $hls_vlnv ${kernel_name}_0

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins ${kernel_name}_0/ap_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins ${kernel_name}_0/ap_rst_n]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/processing_system7_0/M_AXI_GP0" Clk "/processing_system7_0/FCLK_CLK0"} \
    [get_bd_intf_pins ${kernel_name}_0/s_axi_control]

foreach intf {m_axi_gmem0 m_axi_gmem1 m_axi_gmem2 m_axi_gmem3 m_axi_gmem4} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config {Slave "/processing_system7_0/S_AXI_HP0" Clk "/processing_system7_0/FCLK_CLK0"} \
        [get_bd_intf_pins ${kernel_name}_0/$intf]
}

assign_bd_address
validate_bd_design
save_bd_design
make_wrapper -files [get_files [get_property FILE_NAME [current_bd_design]]] -top
add_files -norecurse [glob [file join $project_dir spmv_burst.gen sources_1 bd spmv_burst_bd hdl *wrapper.v]]
update_compile_order -fileset sources_1
write_bd_tcl -force [file join $repo_dir build vivado spmv_burst_bd_export.tcl]
