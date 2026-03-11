using Pkg
Pkg.activate(".")

# using DSP
using Random
using StatsBase
using CSV
using DataFrames
using GLM 
using Statistics
using Plots
using ImageFiltering
using Clustering
using Distances
# using Base.Threads
# using Foresight
using Distributions
# Foresight.set_theme!(foresight(:physics))

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/ISI_KMEAN"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

# 1. Pool a subset of data from all files to find the 9 global centers
println("Phase 1: Finding 9 global centers...")
all_x = Float64[]
all_y = Float64[]

for file in csv_files
    df = CSV.read(file, DataFrame)
    # Take every 10th point to save memory, plenty for finding stable centers
    append!(all_x, df.value1[1:10:end]) 
    append!(all_y, df.value2[1:10:end])
end

global_data = hcat(all_x, all_y)'
R_global = kmeans(global_data, 9) # k=9 for the 9-dot grid
global_centers = R_global.centers
println("Global centers established!")

FigNumber = 1
# Function to assign every x,y coordinate to the nearest global center
function assign_to_centers(x, y, centers)
    n = length(x)
    k_clusters = size(centers, 2)
    assignments = zeros(Int, n)
    
    for i in 1:n
        min_dist = Inf
        best_c = 1
        for c in 1:k_clusters
            # Calculate squared distance to each center
            dist = (x[i] - centers[1, c])^2 + (y[i] - centers[2, c])^2
            if dist < min_dist
                min_dist = dist
                best_c = c
            end
        end
        assignments[i] = best_c
    end
    return assignments
end

# Fixed Hysteresis logic
function extract_stable_fixations(assignments; min_dwell=200)
    # This will hold NamedTuples: (start_idx, stop_idx, cluster_id)
    fixations = [] 
    
    current_state = assignments[1]
    streak_start = 1
    
    for i in 2:lastindex(assignments)
        if assignments[i] != current_state
            streak_length = i - streak_start
            
            # If the streak lasted long enough, log it as a confirmed fixation!
            if streak_length >= min_dwell
                push!(fixations, (start=streak_start, stop=i-1, state=current_state))
            end
            
            # Reset the streak tracker for the new cluster
            current_state = assignments[i]
            streak_start = i
        end
    end
    
    # Catch the final fixation at the very end of the array
    if (length(assignments) - streak_start + 1) >= min_dwell
        push!(fixations, (start=streak_start, stop=length(assignments), state=current_state))
    end
    
    return fixations
end

isi_vec = Float64[]
FigNumber = 1

println("Phase 2: Processing individual files...")
for file in csv_files
    df = CSV.read(file, DataFrame)
    x = df.value1
    y = df.value2
    
    assignments = assign_to_centers(x, y, global_centers)
    
    # 1. Extract only the solid fixations
    # Note: Assuming 1000Hz (dt=0.001), 1000ms is a very long time for a human 
    # visual fixation. 150ms - 300ms is standard. I've lowered it to 200 here.
    fixations = extract_stable_fixations(assignments; min_dwell=1000)

    if length(fixations) > 1
        # 2. Extract Saccade Starts (the frame after a fixation ends)
        sac_starts = [f.stop + 1 for f in fixations[1:end-1]]
        
        # 3. Calculate ISI (Time between the start of consecutive fixations)
        fixation_starts = [f.start for f in fixations]
        isi = diff(fixation_starts)
        
        append!(isi_vec, isi)
        
        # --- Plotting ---
        p4 = plot(x, label="X Position", color=:blue, alpha=0.6)
        plot!(y, label="Y Position", color=:green, alpha=0.6)
        
        # Mark the exact moment the eye leaves the dot to begin searching
        plot!(sac_starts, x[sac_starts], seriestype=:scatter, 
              markersize=4, color=:red, label="Saccade Departure")
              
        # Optional: Mark when the eye finally settles on the new dot
        settle_starts = [f.start for f in fixations[2:end]]
        plot!(settle_starts, x[settle_starts], seriestype=:scatter, 
              markersize=4, color=:black, shape=:star5, label="Target Settled")
        
        plot!(xlims=(10000, 300000), title="Saccade Detection Trace")
        
        outname4 = joinpath(output_dir, splitext(basename(file))[1] * "-SACCADES.pdf")
        savefig(p4, outname4)
    end
end


# timelength over which we plot:
timelength = 120000.
# Precomputing the log normal fit
lISI = log.(isi_vec)
log_σ = std(lISI)
log_peak = median(lISI)
plotting_x = range(0,timelength)
fitted_ln = LogNormal(log_peak,log_σ)

p = histogram(isi_vec,
bins=range(0, timelength, length=60),
normalize=:pdf,
xlabel="Inter-Saccadic Interval (ms)",
# title="File: $(basename(file))",
xlims=(0, timelength),
legend=false)

plot!(plotting_x,pdf.(fitted_ln,plotting_x))

outname = joinpath(output_dir, "Total_ISIs.pdf")
savefig(p, outname)

p3 = plot(isi_vec, xlabel="Index", ylabel="ISI (ms)", title="ISI timeseries")
outname3 = joinpath(output_dir, "ISI_TIMESERIES.pdf")
savefig(p3, outname3)

### Poincaré Plot ###

xs = isi_vec[1:end-1]
ys = isi_vec[2:end]

diff_vec = ys .- xs       # Perpendicular changes
sum_vec = ys .+ xs        # Longitudinal changes

sd1 = std(diff_vec) / sqrt(2)
sd2 = std(sum_vec) / sqrt(2)

println("Rhythm Stability Metrics:")
println("SD1 (Short-term jitter): ", round(sd1, digits=2))
println("SD2 (Long-term drift):   ", round(sd2, digits=2))
println("Ratio (SD2/SD1):         ", round(sd2/sd1, digits=2))
println("Log SD:                  ", round(log_σ, digits=3))

poinc_plot = scatter(xs, ys, 
label="Intervals", 
alpha=0.6, 
markerstrokewidth=0,
color=:blue,
aspect_ratio=:equal, # Important to see true shape
title="Poincaré Plot (Return Map)",
xlabel="Interval t (ms)",
ylabel="Interval t+1 (ms)"
)

# Add Line of Identity (Perfect Rhythm Line)
plot!([minimum(isi_vec), maximum(isi_vec)], [minimum(isi_vec), maximum(isi_vec)], 
label="Line of Identity (y=x)", color=:red, linestyle=:dash)

poinc_outname = joinpath(output_dir, "POINCARE.pdf")
savefig(poinc_plot,poinc_outname)
