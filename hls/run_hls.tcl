set script_dir [file dirname [file normalize [info script]]]
set repo_dir [file dirname $script_dir]

if {![info exists ::env(TOP_NAME)]} {
    error "Set TOP_NAME before invoking run_hls.tcl"
}
if {![info exists ::env(SRC_FILE)]} {
    error "Set SRC_FILE before invoking run_hls.tcl"
}
if {![info exists ::env(TB_FILE)]} {
    error "Set TB_FILE before invoking run_hls.tcl"
}

set top $::env(TOP_NAME)
set src [file normalize $::env(SRC_FILE)]
set tb [file normalize $::env(TB_FILE)]
if {[info exists ::env(PART_NAME)]} {
    set part_name $::env(PART_NAME)
} else {
    set part_name {xc7k70tfbg484-1}
}
set project_dir [file normalize [file join $repo_dir build hls $top]]
set ip_dir [file normalize [file join $repo_dir build ip]]
file mkdir $ip_dir

open_project -reset $project_dir
set_top $top
add_files $src
add_files -tb $tb -cflags "-I[file join $script_dir include] -DTOP_FUNC=$top"
open_solution -reset "solution1" -flow_target vivado
set_part $part_name
create_clock -period 10 -name default
csim_design -clean
csynth_design
export_design -format ip_catalog -output [file join $ip_dir "$top.zip"]
exit
