# Run with:
#   vivado -mode batch -source vivado/create_lr_burst_bd.tcl
#
# This creates a Zynq PS + HLS logistic-regression accelerator block design
# ready for synthesis, implementation, and PYNQ .bit/.hwh export.

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file dirname $script_dir]
set part_name [expr {[info exists ::env(PYNQ_PART)] ? $::env(PYNQ_PART) : "xc7z020clg400-1"}]
set kernel_name [expr {[info exists ::env(LR_KERNEL)] ? $::env(LR_KERNEL) : "lr_train_accel"}]
set vivado_project_dir [file normalize [file join $project_dir build vivado lr_train_accel]]
set required_vivado_version [expr {[info exists ::env(REQUIRED_VIVADO_VERSION)] ? $::env(REQUIRED_VIVADO_VERSION) : "2025.2"}]

if {![string match "${required_vivado_version}*" [version -short]]} {
    error "Incompatible Vivado version [version -short]; expected $required_vivado_version"
}

proc reconnect_bd_pin {driver_pin sink_pin} {
    set existing_nets [get_bd_nets -quiet -of_objects $sink_pin]
    foreach net $existing_nets {
        disconnect_bd_net $net $sink_pin
    }
    connect_bd_net $driver_pin $sink_pin
}

create_project -force lr_train_accel $vivado_project_dir -part $part_name
set_property target_language Verilog [current_project]

set ip_dirs [glob -nocomplain [file join $project_dir build hls $kernel_name solution1 impl ip]]
if {[llength $ip_dirs] == 0} {
    error "No exported HLS IP found for $kernel_name. Run hls/build_hls.ps1 first."
}
if {![file exists [file join [lindex $ip_dirs 0] component.xml]]} {
    error "Exported HLS IP is incomplete: missing component.xml in [lindex $ip_dirs 0]"
}
set_property ip_repo_paths $ip_dirs [current_project]
update_ip_catalog

create_bd_design lr_train_bd
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} \
    [get_bd_cells processing_system7_0]
set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]

set hls_vlnv [lindex [get_ipdefs -all -filter "NAME == $kernel_name"] 0]
if {$hls_vlnv eq ""} {
    error "Could not find IP definition for $kernel_name in $ip_dirs"
}
create_bd_cell -type ip -vlnv $hls_vlnv ${kernel_name}_0

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins ${kernel_name}_0/ap_clk]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/processing_system7_0/M_AXI_GP0" Clk "/processing_system7_0/FCLK_CLK0"} \
    [get_bd_intf_pins ${kernel_name}_0/s_axi_control]

set rst_peripheral [get_bd_pins -quiet rst_ps7_0_50M/peripheral_aresetn]
set rst_interconnect [get_bd_pins -quiet rst_ps7_0_50M/interconnect_aresetn]
if {[llength $rst_peripheral] == 0} {
    set rst_peripheral [get_bd_pins processing_system7_0/FCLK_RESET0_N]
}
if {[llength $rst_interconnect] == 0} {
    set rst_interconnect $rst_peripheral
}
reconnect_bd_pin $rst_peripheral [get_bd_pins ${kernel_name}_0/ap_rst_n]

set hls_m_axi_pins [lsort [get_bd_intf_pins -quiet ${kernel_name}_0/m_axi*]]
if {[llength $hls_m_axi_pins] == 0} {
    error "No AXI master ports found on ${kernel_name}_0"
}

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* axi_mem_interconnect_0
set_property -dict [list CONFIG.NUM_SI [llength $hls_m_axi_pins] CONFIG.NUM_MI {1}] \
    [get_bd_cells axi_mem_interconnect_0]

connect_bd_intf_net [get_bd_intf_pins axi_mem_interconnect_0/M00_AXI] \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]

set axi_index 0
foreach intf_pin $hls_m_axi_pins {
    set si_name [format "S%02d_AXI" $axi_index]
    connect_bd_intf_net $intf_pin [get_bd_intf_pins axi_mem_interconnect_0/$si_name]
    incr axi_index
}

foreach clk_pin [get_bd_pins -quiet axi_mem_interconnect_0/*ACLK] {
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] $clk_pin
}
foreach rst_pin [get_bd_pins -quiet axi_mem_interconnect_0/*ARESETN] {
    if {[string match "*/ARESETN" $rst_pin]} {
        connect_bd_net $rst_interconnect $rst_pin
    } else {
        connect_bd_net $rst_peripheral $rst_pin
    }
}
foreach rst_pin [get_bd_pins -quiet ps7_0_axi_periph/*ARESETN] {
    if {[string match "*/ARESETN" $rst_pin]} {
        reconnect_bd_pin $rst_interconnect $rst_pin
    } else {
        reconnect_bd_pin $rst_peripheral $rst_pin
    }
}

assign_bd_address
validate_bd_design
save_bd_design

make_wrapper -files [get_files [get_property FILE_NAME [current_bd_design]]] -top
add_files -norecurse [glob [file join $vivado_project_dir lr_train_accel.gen sources_1 bd lr_train_bd hdl *wrapper.v]]
update_compile_order -fileset sources_1
write_bd_tcl -force [file join $project_dir build vivado lr_train_bd_export.tcl]
