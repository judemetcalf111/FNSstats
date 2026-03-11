using Pkg
Pkg.activate(".")

using CSV
using SavitzkyGolay
using DSP
using DataFrames
using GLM
using Statistics
using Plots
using Base.Threads

# Load CSV

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/SMSD"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

FigNumber = 1

# --- Define the MAD Function ---
function calculate_smsd(x, y,n)
    N = length(x)
    msd_result = Float64[]
    
    # Loop through time lags (tau) from 1 up to half the trajectory length
    # (Going beyond N/2 usually makes the statistics too noisy)
    for tau in 1:n
        # Create shifted arrays to calculate displacement
        # x(t + tau) - x(t)
        dx = x[1+tau:end] .- x[1:end-tau]
        dy = y[1+tau:end] .- y[1:end-tau]
        
        # Squared displacement: dx^2 + dy^2
        MSD = mean((dx).^2 .+ (dy).^2)
        
        # Mean of the squared displacements for this lag
        push!(msd_result, MSD)
    end
    
    return msd_result
end

for file in csv_files
    # Read CSV into DataFrame
    df = CSV.read(file, DataFrame)

    # Extract x and y
    x = df.value1
    y = df.value2

    # 1. Define Sampling Frequency
    dt = 0.001
    sample_freq = 1/dt

    # 2. Define Response (Place 'fs' HERE)
    # If you omit 'fs', 0.5 is treated as normalized frequency (half-Nyquist).
    # By including 'fs', 0.5 is treated as 0.5 Hz.
    responsetype = Lowpass(10; fs=sample_freq)

    # 3. Define Design Method
    designmethod = Butterworth(4)

    # 4. Construct Filter (Do this once!)
    # It is inefficient to call digitalfilter() every time you filter a vector.
    my_filter = digitalfilter(responsetype, designmethod)

    # 5. Apply Filter
    x_sm = filt(my_filter, x)
    y_sm = filt(my_filter, y)

    window_size = 61 # approx 6ms
    x_sg = savitzky_golay(x, window_size, 2, rate=1/dt).y
    y_sg = savitzky_golay(y, window_size, 2, rate=1/dt).y
    
    plot(x[1:10000])
    plot!(x_sm[1:10000])
    plot!(x_sg[1:10000])

    outname = joinpath(output_dir, splitext(basename(file))[1] * "-SmoothPath" * ".pdf")
    # Save figure
    savefig(outname)

    n = 20000
    t = 0.001 * collect(1:n)
    xtest = x[1:end-n]
    m = length(xtest)

    # Run and print
    msd_values = calculate_smsd(x_sg, y_sg, n)

    println("Calculation Complete.")
    println("First 5 MAD values:")
    for i in 1:min(5, length(msd_values))
        println("Lag $i: $(msd_values[i])")
    end

    # Update DataFrame Logic
    powerlaw_n = 100
    df = DataFrame(X = log.(t[1:powerlaw_n]), Y = log.(msd_values[1:powerlaw_n]))

    model = lm(@formula(Y ~ X), df)
    Coeffs = GLM.coef(model)
    
    plot(
        t,
        msd_values,
        # label = "Squared MAD",       # Renamed from Means
        linewidth = 8,
        linestyle = :solid,
        color = :Orange,
        xscale = :log10,
        yscale = :log10,
        grid = true,
        size = (800, 600),           # Increases the base resolution canvas
        dpi = 1200,                  # High resolution for crisp lines and text
        guidefontsize = 30,          # Increases axis label font size (xlabel, ylabel)
        tickfontsize = 30,           # Increases axis number/tick font size
        legend = false,
        framestyle = :box,           # Adds a clean border around the entire plot
        margin = 5Plots.mm           # Prevents larger labels from being cut off
    )

    plot!(
        t[1:powerlaw_n],
        (t[1:powerlaw_n].^Coeffs[2]) .* (exp(Coeffs[1])), # slightly cleaner exp() syntax
        # label = "Trendline",
        linewidth = 8,
        linestyle = :dash,
        legend = false,
        color = :Black
    )

    # Update axis labels for MAD
    plot!(xlabel = "Time", ylabel = "Squared MAD ⟨r⟩²", legend = false)

    endpoint = 0.02

    tpos = endpoint                 # choose a spot
    ypos = (endpoint.^Coeffs[2]) .* (ℯ.^Coeffs[1]) .* (10^(-1))   # y-value to match scale
    eqn_string = "t^$(round(Coeffs[2], digits=4))"

    annotate!((tpos, ypos, text(eqn_string, :Black, 30, :left)))

    outname = joinpath(output_dir, splitext(basename(file))[1] * "-SMAD" * ".pdf")

    # Save figure

    savefig(outname)

    println("Fig number: $FigNumber" )
    global FigNumber = FigNumber + 1
end
