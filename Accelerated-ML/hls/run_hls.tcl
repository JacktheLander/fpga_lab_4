set script_dir [file dirname [file normalize [info script]]]
set project_dir [file dirname $script_dir]
set top_name [expr {[info exists ::env(TOP_NAME)] ? $::env(TOP_NAME) : "lr_train_accel"}]
set part_name [expr {[info exists ::env(PART_NAME)] ? $::env(PART_NAME) : "xc7z020clg400-1"}]
set clock_period [expr {[info exists ::env(CLOCK_PERIOD)] ? $::env(CLOCK_PERIOD) : "10"}]
set csim_only [expr {[info exists ::env(HLS_CSIM_ONLY)] && $::env(HLS_CSIM_ONLY) ne "0"}]

set include_dir [file normalize [file join $script_dir include]]
set src_file [file normalize [file join $script_dir src lr_train_accel.cpp]]
set tb_file [file normalize [file join $script_dir tb tb_lr_train_accel.cpp]]
set build_dir [file normalize [file join $project_dir build hls $top_name]]
set ip_dir [file normalize [file join $project_dir build ip]]
file mkdir $ip_dir

open_project -reset $build_dir
set_top $top_name
add_files $src_file -cflags "-I$include_dir -std=c++14"
add_files -tb $tb_file -cflags "-I$include_dir -std=c++14"
open_solution -reset "solution1" -flow_target vivado
set_part $part_name
create_clock -period $clock_period -name default

csim_design -clean

if {!$csim_only} {
    csynth_design
    export_design -format ip_catalog -output [file join $ip_dir "$top_name.zip"]
}

exit
