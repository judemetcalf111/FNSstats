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
using ProgressBars
# using Base.Threads
# using Foresight
using Distributions
# Foresight.set_theme!(foresight(:physics))

# Set input and output folders
data_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"

println("Please enter the directory containing your files: ") # Prompt the user for input
dir_name = readline()                   # Read the input and store it in the 'name' variable
input_dir = joinpath(data_dir, dir_name) # Construct the full path to the input directory
number_of_files = length(filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true)))
# try 
# catch
#     println("Error occurred while reading directory: $dir_name")
#     exit(1)
# end

println("Very nice, there are $number_of_files files in the directory you entered: $dir_name") # Print the input back to the user

output_dir = joinpath(data_dir, dir_name * "_ISIs")

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

function calculate_isi(x, y; timeout_smoother=.001, timeout=0.001, λ=20., dt=0.001, min_dur_steps=3)
    timeout_steps = Int(ceil(timeout/dt))
    timeout_smoother_steps = Int(ceil(timeout_smoother/dt))
    
    # smooth the velocities with just a uniform averaging of position
    # (physically motivated by an averaging across time? Muscles don't contract instantly, neural lag...)
    # Since we are looking for macro-saccades here, this is basically a low-pass filter, 
    # strength determined by timeout parameter, which is the window of time averaged over, over which microsaccades fade away
    # function rolling_mean(vec, n)
    #     return [mean(@view vec[max(1, i-n+1):i]) for i in 1:length(vec)]
    # end

    function gaussian_smooth(vec, σ)
        # Maths of the convolution is handled by ImageFiltering.jl, we just need to specify the kernel and sd
        return imfilter(vec, Kernel.gaussian((σ,)))
    end

    vx = [0.0; diff(x)] ./ dt
    vy = [0.0; diff(y)] ./ dt

    jumps = sqrt.(vx.^2 .+ vy.^2)

    v = gaussian_smooth(jumps, timeout_smoother_steps)

    # threshold
    msd = median(abs.(v .- median(v))) / 0.6745
    threshold = λ * msd
    is_saccade = v .> threshold

    # Duration of saccade and making sure we don't double count the same saccade
    starts_timed = Int[]
    last_start = - timeout_steps # Initialize to allow the first detection
    
    i::Int = 1

    while i < length(is_saccade)
        if is_saccade[i]
            # Find how long this specific "high velocity" event lasts
            j = i
            while j <= length(is_saccade) && is_saccade[j]
                j += 1
            end
            event_duration = j - i
            
            # Long enough && Enough time since last one
            # Warning: FNS gives saccades very quickly, on the order of 1 timeout_steps
            # so set min_dur_steps to between 1 and 4, above that and saccades vanish
            if event_duration >= min_dur_steps && (i - last_start) > timeout_steps
                push!(starts_timed, i)
                last_start = i
            end
            
            # Skip to the end of this velocity peak
            i = j
        else
            i += 1
        end
    end

    # ISI in ms
    isi = diff(starts_timed) .* (dt * 1000) 

    return (starts_timed, isi) 
end

# loop through csv files in /datadir
isi_total = Int[]

for file in ProgressBar(csv_files)
    df = CSV.read(file, DataFrame)
    x = df.value1
    y = df.value2
    dt = 0.001

    (_, isi) = calculate_isi(x, y; timeout_smoother=0.02, timeout=0.006, λ=2., dt=dt, min_dur_steps=4)

    append!(isi_total,isi)
end

df_isi = DataFrame(ISI = isi_total)

CSV.write(joinpath(output_dir,"ISI.csv"), df_isi)

if length(isi_total) > 1
    # timelength over which we plot:
    timelength = 2500.
    # Precomputing the log normal fit
    lISI = log.(isi_total)
    log_σ = std(lISI)
    log_peak = median(lISI)
    plotting_x = range(0,timelength)
    fitted_ln = LogNormal(log_peak,log_σ)

    p = histogram(isi_total,
                bins=range(0, timelength, length=40),
                normalize=:pdf,
                xlabel="Inter-Saccadic Interval (ms)",
                xlims=(0, timelength),
                legend=false)

    plot!(plotting_x,pdf.(fitted_ln,plotting_x))

    outname = joinpath(output_dir,"ISI_TRANS.pdf")
    savefig(p, outname)

    lags = 0:length(isi_total)-1
    acf_values = autocor(isi_total, lags)

    p2 = plot(lags, acf_values, xlims=(0,40), ylims=(-0.1,0.2), xlabel="Lag", ylabel="Autocorrelation", title="Autocorrelation of ISIs")
    outname2 = joinpath(output_dir, "ISI_ACF.pdf")

    shuffled_acf = autocor(shuffle(isi_total), lags)
    plot!(lags, shuffled_acf, label="Shuffled ISIs", color=:red, linestyle=:dash)
    savefig(p2, outname2)

    p3 = plot(isi_total, xlabel="Index", ylabel="ISI (ms)", title="ISI timeseries")
    outname3 = joinpath(output_dir, "ISI_TIMESERIES.pdf")
    savefig(p3, outname3)

    p4 = plot((x+y)./2)
    plot!(sac_starts, (x+y)[sac_starts]./2, seriestype=:scatter, linewidth=1, xlims=(10000, 30000), ylims=(-10, 10), label="Saccade Starts", color=:red)
    outname4 = joinpath(output_dir, "SACCADES.pdf")
    savefig(p4, outname4)
end


