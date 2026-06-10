# Run through vivado/export_pynq_artifacts.ps1 after create_lr_burst_bd.tcl.
#
# Required environment variables:
#   LR_PROJECT_XPR     Path to lr_train_accel.xpr.
#   LR_ARTIFACT_NAME   Basename for exported .bit/.hwh.
#   LR_ARTIFACT_DIR    Destination directory.

foreach required {LR_PROJECT_XPR LR_ARTIFACT_NAME LR_ARTIFACT_DIR} {
    if {![info exists ::env($required)] || $::env($required) eq ""} {
        error "Missing required environment variable $required"
    }
}

set xpr_path [file normalize $::env(LR_PROJECT_XPR)]
set artifact_name $::env(LR_ARTIFACT_NAME)
set artifact_dir [file normalize $::env(LR_ARTIFACT_DIR)]
file mkdir $artifact_dir

if {![file exists $xpr_path]} {
    error "Vivado project not found: $xpr_path"
}

open_project $xpr_path
set project_dir [get_property DIRECTORY [current_project]]

set bd_files [get_files -quiet -filter {FILE_TYPE == "Block Designs"}]
if {[llength $bd_files] == 0} {
    set bd_files [glob -nocomplain [file join $project_dir *.srcs sources_1 bd * *.bd]]
}
if {[llength $bd_files] == 0} {
    error "No block design found in $xpr_path"
}
set bd_file [lindex $bd_files 0]

open_bd_design $bd_file
validate_bd_design
save_bd_design
generate_target all [get_files $bd_file]
update_compile_order -fileset sources_1

reset_run synth_1
launch_runs synth_1 -jobs 6
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
if {![regexp {Complete} $synth_status]} {
    error "synth_1 did not complete. Status: $synth_status"
}

launch_runs impl_1 -to_step write_bitstream -jobs 6
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
if {![regexp {Complete} $impl_status]} {
    error "impl_1 did not complete. Status: $impl_status"
}

set bit_files [glob -nocomplain -types f [file join $project_dir *.runs impl_1 *.bit]]
if {[llength $bit_files] == 0} {
    error "No bitstream found under $project_dir/*.runs/impl_1"
}
set bit_file [lindex $bit_files 0]

set hwh_files [glob -nocomplain -types f [file join $project_dir *.gen sources_1 bd * hw_handoff *.hwh]]
if {[llength $hwh_files] == 0} {
    set hwh_files [glob -nocomplain -types f [file join $project_dir *.srcs sources_1 bd * hw_handoff *.hwh]]
}
if {[llength $hwh_files] == 0} {
    error "No HWH handoff file found under $project_dir"
}
set hwh_file [lindex $hwh_files 0]

set out_bit [file join $artifact_dir "${artifact_name}.bit"]
set out_hwh [file join $artifact_dir "${artifact_name}.hwh"]
file copy -force $bit_file $out_bit
file copy -force $hwh_file $out_hwh

puts "Exported PYNQ artifacts:"
puts "  $out_bit"
puts "  $out_hwh"
