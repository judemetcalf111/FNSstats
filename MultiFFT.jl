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
using SavitzkyGolay

# Set default plotting visuals

default(
    fontfamily = "Computer Modern",    # Nice font, renders a minus sign
    titlefontsize = 24,
    guidefontsize = 22,          
    tickfontsize = 18,
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

    dt = 0.0001
    fs = 1.0 / (dt * 0.023) 
    
    df_first = CSV.read(csv_files[1], DataFrame)
    N_points = length(df_first.value1)
    n_window = div(2 * N_points, 3)

    # Run a dummy periodogram to let DSP.jl determine the FFT length
    dummy_x = df_first.value1 .- mean(df_first.value1)
    dummy_pgram = welch_pgram(dummy_x, n_window; fs=fs, onesided=true, window=hanning)
    N_freqs = length(power(dummy_pgram)) # This will perfectly capture the 134401 length

    W_X_matrix = zeros(Float64, N_freqs, tot)
    W_Y_matrix = zeros(Float64, N_freqs, tot)
    W_V_matrix = zeros(Float64, N_freqs, tot)    
    freqs_out = Vector{Float64}()
    freqs_v_out = Vector{Float64}()
    
    valid_files = 0

    for (i, file) in enumerate(csv_files)
        df = CSV.read(file, DataFrame)
        
        x = df.value1 .- mean(df.value1)
        y = df.value2 .- mean(df.value2)

        # True velocity is dx/dt. Length becomes N-1
        vx = diff(x) ./ dt
        vy = diff(y) ./ dt
        v = sqrt.(vx.^2 .+ vy.^2)

        v = v .- mean(v) # Subtracting the DC offset

        window_size = 501

        v = savitzky_golay(v, window_size, 2, rate=1/dt).y

        # Checking `v` is safest since it's the shortest array (N-1)
        if length(v) < n_window
            println("Skipping file $i: Not enough data points for window size $n_window")
            continue
        end

        # Compute periodograms
        pgram_x = welch_pgram(x, n_window; fs=fs, onesided=true, window=hanning)
        pgram_y = welch_pgram(y, n_window; fs=fs, onesided=true, window=hanning)
        pgram_v = welch_pgram(v, n_window; fs=fs, onesided=true, window=hanning)

        # Capture frequency bins from the first valid file
        if valid_files == 0
            freqs_out = freq(pgram_x)
            freqs_v_out = freq(pgram_v)
        end

        # Increment valid files counter and store in the matrix
        valid_files += 1
        
        W_X_matrix[:, valid_files] = power(pgram_x)
        W_Y_matrix[:, valid_files] = power(pgram_y)
        W_V_matrix[:, valid_files] = power(pgram_v)
        
        println("Finished with file $i of $tot")
    end

    if valid_files == 0
        println("Error: No files were long enough to process.")
        return
    end

    W_X_matrix = W_X_matrix[:, 1:valid_files]
    W_Y_matrix = W_Y_matrix[:, 1:valid_files]
    W_V_matrix = W_V_matrix[:, 1:valid_files]

    W_X_avg = vec(mean(W_X_matrix, dims=2))
    W_Y_avg = vec(mean(W_Y_matrix, dims=2))
    W_V_avg = vec(mean(W_V_matrix, dims=2))

    W_X_sem = vec(std(W_X_matrix, dims=2)) ./ sqrt(valid_files)
    W_Y_sem = vec(std(W_Y_matrix, dims=2)) ./ sqrt(valid_files)
    W_V_sem = vec(std(W_V_matrix, dims=2)) ./ sqrt(valid_files)

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

    downsample(v::Vector{Float64}) = v #[1:100:end] # Define a downsampler to be able to view the error bars

    # --- LINEAR PLOT ---
    plot_pos_lin = plot(
        freqs_out[1:1000], W_X_avg[1:1000], # Notice trimming so that the ribbons don't get a "Polygon too complex for filling." error
        # ribbon = 3 .* W_X_sem,
        # label = "X-Position PSD", 
        linewidth = 4, 
        color = :steelblue,
        xlabel = "Frequency (Hz)",
        ylabel = "Power",
        xlims = (0, 10) # Capped at Nyquist, starts at 0
    )
    plot!(freqs_out[1:1000], W_Y_avg[1:1000], # Here, we downsample.
        # label = "Y-Position PSD",
        # ribbon = 3 .* W_Y_sem,
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
        downsample(freqs_log), downsample(psd_x_log),  # Downsampling, since we only care about the tail...
        # ribbon = 3 .* downsample(W_X_sem),
        # label = "X-Position PSD",
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
        downsample(freqs_log), downsample(psd_y_log),
        # ribbon = 3 .* downsample(W_Y_sem),
        # label = "Y-Position PSD",
        linewidth = 4,
        color = :darkorange
    )
    savefig(plot_pos_log, joinpath(output_dir, "Welch_x_Log.pdf"))

    println("Successfully generated Linear and Logarithmic plots!")

    # --- LINEAR PLOT ---
    plot_v_lin = plot(
        freqs_v_out[2:1000], W_V_avg[2:1000],
        # ribbon = 3 .* W_V_sem,
        # label = "Velocity PSD (Linear)", 
        linewidth = 4, 
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
        downsample(freqs_v_log), downsample(psd_v_log),
        # label = "Velocity PSD (Log-Log)", 
        # ribbon = 3 .* downsample(W_V_sem),
        linewidth = 4, 
        color = :darkorange,
        xscale = :log10, 
        yscale = :log10,
        xlabel = "Frequency (Hz)",
        ylabel = "Power",
        # Start xlims slightly above 0, e.g., your lowest non-zero frequency
        xlims = (minimum(freqs_log), 2000)
    )
    savefig(plot_v_log, joinpath(output_dir, "Welch_v_Log.pdf"))

    println("Successfully generated Linear and Logarithmic plots!")
end

calculate_walker_psd(input_dir,output_dir)
