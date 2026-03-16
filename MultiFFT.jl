using Pkg
Pkg.activate(".")

using DSP
using CSV
using DataFrames
using GLM
using Statistics
using Plots
using Plots.Measures
using Base.Threads

# Set default plotting visuals

default(
    fontfamily = "Computer Modern",    # Nice font, renders a minus sign
    titlefontsize = 24,
    guidefontsize = 18,          
    tickfontsize = 10,
    legendfontsize = 18,
    grid = false,                # Removes visual clutter for smaller plots
    framestyle = :box,           # Professional enclosed bounding box
    dpi = 300,                   # High resolution for PDF export
    margin = 8Plots.mm           # Fixing the label cropping...
)

# Load CSV
# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/Welch"

function calculate_walker_psd(input_dir, output_dir)
    # Make sure output folder exists
    isdir(output_dir) || mkdir(output_dir)

    # Get all .csv files in the input directory
    csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
    tot = length(csv_files)
    
    if tot == 0
        println("No CSV files found in $input_dir")
        return
    end

    dt = 0.001
    fs = 1.0 / (dt * 0.023) # 0.023 is my magic number to rescale from FNS time to real time
    
    # Use a fixed window size (using 4 windows here to ensure the theta frequencies are included)
    # This guarantees frequency bins align perfectly across all files
    df_first = CSV.read(csv_files[1], DataFrame)
    N_points = length(df_first.value1)
    n_window = div(2 * N_points, 3)

    # Initialize variables to hold sums
    W_X_sum = nothing
    W_Y_sum = nothing
    W_V_sum = nothing
    freqs_out = nothing
    freqs_v_out = nothing

    for (i, file) in enumerate(csv_files)
        df = CSV.read(file, DataFrame)
        
        x = df.value1 .- mean(df.value1)        # Subtract the DC offset, just to ensure that all trials have the same vanishing 0 frequency, so as not to mess up FFT results
        y = df.value2 .- mean(df.value2)        # see above

        # True velocity is dx/dt. Note: diff() makes the array length N-1, 
        # keep in mind when plotting if you get off-by-one errors
        vx = diff(x) ./ dt
        vy = diff(y) ./ dt

        v = sqrt.(vx.^2 .+ vy.^2)

        # Compute periodograms
        pgram_x = welch_pgram(x, n_window; fs=fs, onesided=true, window=hanning)
        pgram_y = welch_pgram(y, n_window; fs=fs, onesided=true, window=hanning)
        pgram_v = welch_pgram(v, n_window; fs=fs, onesided=true, window=hanning)

        if i == 1
            # First iteration: initialize the accumulators
            freqs_out = freq(pgram_x)
            freqs_v_out = freq(pgram_v)
            
            W_X_sum = power(pgram_x)
            W_Y_sum = power(pgram_y)
            # Total Power is the sum of X and Y powers
            W_V_sum = power(pgram_v)
        else
            # Subsequent iterations: add to accumulators
            W_X_sum .+= power(pgram_x)
            W_Y_sum .+= power(pgram_y)
            W_V_sum .+= power(pgram_v)
        end
    end

    # Calculate the ensemble average
    W_X_avg = W_X_sum ./ tot
    W_Y_avg = W_Y_sum ./ tot
    W_V_avg = W_V_sum ./ tot

    # Save to CSV
    csv_outpath = joinpath(output_dir, "Welch_arrays.csv")
    
    # Note: If x and vx have different window logic, their freq vectors 
    # might differ slightly in length. Better to save them safely.
    CSV.write(csv_outpath, DataFrame(
        Freqs_X = freqs_out, 
        Welch_X = W_X_avg,
        Welch_Y = W_Y_avg
    ))

    println("Saved X Welch Periodograms to $csv_outpath")

    csv_outpath = joinpath(output_dir, "Welch_arrays.csv")
    
    # Note: If x and vx have different window logic, their freq vectors 
    # might differ slightly in length. Better to save them safely.
    CSV.write(csv_outpath, DataFrame(
        Freqs_V = freqs_v_out, 
        Welch_V = W_V_avg
    ))

    println("Saved V Welch Periodograms to $csv_outpath")

    # --- LINEAR PLOT ---
    plot_pos_lin = plot(
        freqs_out, W_X_avg,
        label = "X-Position PSD", 
        linewidth = 4, 
        color = :steelblue,
        xlabel = "Frequency (Hz)",
        ylabel = "Power",
        xlims = (0, 10) # Capped at Nyquist, starts at 0
    )
    plot!(freqs_out, W_Y_avg,
        label = "Y-Position PSD",
        linewidth = 4,
        color = :darkorange,
        xlims = (0, 10)
    )

    savefig(plot_pos_lin, joinpath(output_dir, "Welch_x_Linear.pdf"))


    # --- LOGARITHMIC PLOT ---
    # We use [2:end] to drop the 0.0 Hz DC component and avoid the log(0) warning
    freqs_log = freqs_out[2:end]
    psd_x_log = W_X_avg[2:end]
    psd_y_log = W_Y_avg[2:end]

    plot_pos_log = plot(
        freqs_log, psd_x_log,
        label = "X-Position PSD",
        linewidth = 4, 
        color = :steelblue,
        xscale = :log10, 
        yscale = :log10,
        xlabel = "Frequency (Hz)",
        ylabel = "Power",
        # Start xlims slightly above 0, e.g., your lowest non-zero frequency
        xlims = (minimum(freqs_log), 20000) 
    )

    plot!(
        freqs_log, psd_y_log,
        label = "Y-Position PSD",
        linewidth = 4,
        color = :darkorange
    )
    savefig(plot_pos_log, joinpath(output_dir, "Welch_x_Log.pdf"))

    println("Successfully generated Linear and Logarithmic plots!")

    # --- LINEAR PLOT ---
    plot_v_lin = plot(
        freqs_v_out, W_V_avg,
        label = "Velocity PSD (Linear)", 
        linewidth = 2, 
        color = :steelblue,
        xlabel = "Frequency (Hz)",
        ylabel = "Power",
        xlims = (0, 10) # Capped at Nyquist, starts at 0
    )
    savefig(plot_v_lin, joinpath(output_dir, "Welch_v_Linear.pdf"))

    # We use [2:end] to drop the 0 Hz DC component and avoid the log(0) warning
    freqs_v_log = freqs_v_out[2:end]
    psd_v_log = W_V_avg[2:end]

    plot_v_log = plot(
        freqs_v_log, psd_v_log,
        label = "Velocity PSD (Log-Log)", 
        linewidth = 2, 
        color = :darkorange,
        xscale = :log10, 
        yscale = :log10,
        xlabel = "Frequency (Hz)",
        ylabel = "Power",
        # Start xlims slightly above 0, e.g., your lowest non-zero frequency
        xlims = (minimum(freqs_log), 20000) 
    )
    savefig(plot_v_log, joinpath(output_dir, "Welch_v_Log.pdf"))

    println("Successfully generated Linear and Logarithmic plots!")
end

calculate_walker_psd(input_dir,output_dir)
