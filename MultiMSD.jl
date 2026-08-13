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
using FFTW

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/MSD_Parallel"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

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

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

FigNumber = 1

# Helper function to compute 1D MSD instantly using the Fast Correlation Algorithm
function compute_fast_msd(x::Vector{Float64}, max_lag::Int)
    N = length(x)
    
    # FFT for the uncentered cross-correlation (the -2*x*y term)
    # We pad the array to the next power of 2 to optimize the FFT speed
    Npad = nextpow(2, 2N - 1)
    X_fft = fft(vcat(x, zeros(Npad - N)))
    S_x = ifft(X_fft .* conj.(X_fft))
    corr = real.(S_x[1:max_lag+1])
    
    # Cumulative sum for the squared terms
    D = x.^2
    CS = vcat([0.0], cumsum(D)) # Prepend 0 to make boundary math easier
    
    msd = zeros(Float64, max_lag + 1)
    
    @inbounds for k in 0:max_lag
        # Instantly calculate the sum of squares using the cumulative sum array
        S_sq = (CS[N+1] - CS[k+1]) + (CS[N-k+1] - CS[1])
        
        # Combine everything according to the expanded MSD formula
        msd[k+1] = (S_sq - 2 * corr[k+1]) / (N - k)
    end
    
    return msd
end

function calculate_filtered_msd(input_dir, output_dir; K_t::Float64 = 1.5, dt::Float64 = 0.0001, cutoff_hz::Float64 = 50.0)
    isdir(output_dir) || mkdir(output_dir)
    
    csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
    tot = length(csv_files)
    
    if tot == 0
        println("No CSV files found.")
        return
    end
    
    max_lag = ceil(Int, K_t / dt)
    
    # Design a 4th-order Butterworth Low-Pass Filter
    # fs is the sampling frequency (1 / dt)
    lp_filter = digitalfilter(Lowpass(cutoff_hz; fs=1.0/dt), Butterworth(4))
    
    sum_msd_x = zeros(Float64, max_lag + 1)
    sum_msd_y = zeros(Float64, max_lag + 1)
    valid_files = 0
    
    println("Calculating filtered fast MSD up to lag $max_lag...")
    
    for (i, file) in enumerate(csv_files)
        df = CSV.read(file, DataFrame)
        
        x_raw = Vector{Float64}(df.value1)
        y_raw = Vector{Float64}(df.value2)
        
        if length(x_raw) <= max_lag
            println("Skipping file $i: Not enough data points.")
            continue
        end
        
        # Apply the Zero-Phase filter (filtfilt) to remove high-frequency noise without shifting time
        x_filtered = filtfilt(lp_filter, x_raw)
        y_filtered = filtfilt(lp_filter, y_raw)
        
        # Calculate MSDs using the O(N log N) helper function
        msd_x_chunk = compute_fast_msd(x_filtered, max_lag)
        msd_y_chunk = compute_fast_msd(y_filtered, max_lag)
        
        sum_msd_x .+= msd_x_chunk
        sum_msd_y .+= msd_y_chunk
        valid_files += 1
        
        println("Finished with file $i of $tot")
    end
    
    # Average the MSDs across all chunks
    final_msd_x = sum_msd_x ./ valid_files
    final_msd_y = sum_msd_y ./ valid_files
    
    # Total 2D MSD is just the sum of the independent 1D MSDs
    final_msd_total = final_msd_x .+ final_msd_y
    
    lag_times = range(0, K_t, length=length(final_msd_x))
    
    # Save to CSV
    out_df = DataFrame(Time = lag_times, MSD_X = final_msd_x, MSD_Y = final_msd_y, MSD_Total = final_msd_total)
    csv_outpath = joinpath(output_dir, "MSD_Filtered_Results.csv")
    CSV.write(csv_outpath, out_df)
    
    println("Saved rapid MSD results to $csv_outpath")
    
    # --- UPDATE PLOT LOGIC ---
    powerlaw_n = 100
    df = DataFrame(X = log.(lag_times[2:powerlaw_n]), Y = log.(final_msd_total[2:powerlaw_n]))

    model = lm(@formula(Y ~ X), df)
    Coeffs = GLM.coef(model)
    
    plot_msd_total = plot(lag_times[2:end] * 1000, final_msd_total[2:end],
        linewidth = 8,
        linestyle = :solid,
        color = :steelblue,
        xscale = :log10,
        yscale = :log10,
        grid = true,
        size = (800, 600),           
        dpi = 1200,                  
        guidefontsize = 30,          
        tickfontsize = 30,           
        legend = false,
        framestyle = :box,           
        margin = 5Plots.mm           
    )
    
    # Trendline Plot (Start at index 2 to avoid log(0), and scale X by 1000)
    plot!(
        lag_times[2:powerlaw_n] * 1000,
        (lag_times[2:powerlaw_n].^Coeffs[2]) .* (exp(Coeffs[1])), 
        linewidth = 8,
        linestyle = :dash,
        legend = false,
        color = :black
    )
    
    plot!(xlabel = "Time (ms)", ylabel = "MSD <r^2>", legend = false)
    
    endpoint = 0.02
    tpos = endpoint * 1000 # Matched to the * 1000 scale
    ypos = (endpoint.^Coeffs[2]) .* (exp(Coeffs[1])) .* (10^(-1))
    eqn_string = "t^$(round(Coeffs[2], digits=4))"
    
    annotate!((tpos, ypos, text(eqn_string, :black, 30, :left)))

    savefig(plot_msd_total, joinpath(output_dir, "MSD_Final.pdf"))
    println("Saved rendered plot to MSD_Final.pdf")
end

calculate_filtered_msd(input_dir, output_dir; K_t=3., dt=0.0001*0.023, cutoff_hz=5000.)
