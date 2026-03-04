using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using GLM 
using Statistics
using Plots
using Base.Threads
using Foresight
Foresight.set_theme!(foresight(:physics))

# Load CSV

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/ISI"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

FigNumber = 1

# --- 2. Define the MAD Function ---
function calculate_isi(x, y; sac_thresh=1.0, timeout=0.006, dt=0.001)
    isi_series = Float64[]
    isi = Float64[]

    timeout_steps = ceil(timeout/dt)

    jumps = sqrt.((x[2:end] .- x[1:end-1]).^2 + (y[2:end] .- y[1:end-1]).^2)
    
    if sac_thresh < .001 || timeout_steps < 1
        error("The sac_thresh variable is set at ", sac_thresh, " the ISI function is not informative")
    end

    # Find the first saccade
    first_sac = findfirst(jump -> jump > sac_thresh, jumps)
    
    # Handle case where no saccade is found
    if first_sac === nothing
        return (isi_series, isi)
    end

    push!(isi_series, first_sac)
    # The first saccade doesn't have an "inter-saccadic interval" 
    # unless you count from time zero. Usually, isi starts from the second sac.

    for (idx, jump) in enumerate(jumps)
        # Check if this is a saccade and if we are past the timeout from the LAST saccade
        if (jump > sac_thresh) && (idx - isi_series[end] > timeout_steps)
            current_interval = idx - isi_series[end]
            push!(isi, current_interval - timeout_steps)  # Subtract timeout to get the "true" ISI
            push!(isi_series, idx)
        end
    end
    
    return (isi_series, isi)
end

for file in csv_files
    # Read CSV into DataFrame
    df = CSV.read(file, DataFrame)

    # Extract the columns (x = value1, y = value2)
    x = df.value1
    y = df.value2

    dt = 0.001 # This is standard across all my experiments

    # Define parameters for calculating the ISI
    sac_thresh = 5.
    timeout = 0.1

    # --- 3. Run and Print ---
    (isi_series, isi) = calculate_isi(x, y, sac_thresh=sac_thresh, timeout=timeout, dt=dt)

    println("Calculation Complete.")
    println("First 5 ISI values:")
    for i in 1:min(5, length(isi))
        println("Lag $i: $(isi[i])")
    end

    if length(isi) > 5
        histogram(isi)
    end

    # # Update DataFrame Logic
    # df = DataFrame(X = log.(t), Y = log.(mad_values))

    # model = lm(@formula(Y ~ X), df)
    # Coeffs = GLM.coef(model)
    
    # plot(
    #     t,
    #     mad_values,
    #     label = "Squared MAD",             # Renamed from Means
    #     linewidth = 5,
    #     linestyle = :solid,
    #     color = :Orange,
    #     xscale = :log10,
    #     yscale = :log10,
    #     grid = true,
    #     size = (800, 600),           # Increases the base resolution canvas
    #     dpi = 1200,                   # High resolution for crisp lines and text
    #     guidefontsize = 20,          # Increases axis label font size (xlabel, ylabel)
    #     tickfontsize = 12,           # Increases axis number/tick font size
    #     legendfontsize = 20,         # Increases legend text size
    #     framestyle = :box,           # Adds a clean border around the entire plot
    #     margin = 5Plots.mm           # Prevents larger labels from being cut off
    # )

    # plot!(
    #     t,
    #     (t.^Coeffs[2]) .* (exp(Coeffs[1])), # slightly cleaner exp() syntax
    #     label = "Trendline",
    #     linewidth = 3,
    #     linestyle = :dash,
    #     color = :Black
    # )

    # # Update axis labels for MAD
    # plot!(xlabel = "Time", ylabel = "Squared MAD ⟨r⟩²", legend = :outerright)

    # endpoint = 0.005

    # tpos = endpoint                 # choose a spot
    # ypos = (endpoint.^Coeffs[2]) .* (ℯ.^Coeffs[1]) .* (10^2)   # y-value to match scale
    # eqn_string = "⟨r⟩ ~ t^$(round(Coeffs[2], digits=3))"


    # annotate!((tpos, ypos, text(eqn_string, :Black, 16, :left)))

    outname = joinpath(output_dir, splitext(basename(file))[1] * "-ISI" * ".pdf")

    # Save figure

    savefig(outname)

    println("Fig number: $FigNumber" )
    global FigNumber = FigNumber + 1
end
