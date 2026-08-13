using Pkg
Pkg.activate(".")

# using DSP
using SavitzkyGolay
using Random
using StatsBase
using CSV
using DataFrames
using GLM
using Statistics
using Plots
using ImageFiltering
using Distributions

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/data"
output_dir  = "/Users/chardiol/Desktop/Theory of Brain/Julian_Plotting/ISI_Parallel/"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

FigNumber = 1

default(
    fontfamily = "Computer Modern",    # Nice font, renders a minus sign
    titlefontsize = 24,
    guidefontsize = 18,          
    tickfontsize = 20,
    legendfontsize = 16,
    grid = false,                # Removes visual clutter for smaller plots
    framestyle = :box,           # Professional enclosed bounding box
    dpi = 300,                   # High resolution for PDF export
    margin = 8Plots.mm,          # Fixing the label cropping...
    legend = :topright,
)

function get_sg_velocity(x, y, window_size, dt)

    # Ensure odd:
    if window_size % 2 == 0
        window_size -= 1
    end    
    poly_order = 3
    sg_x = savitzky_golay(x, window_size, poly_order, deriv=1, rate=1/dt)
    sg_y = savitzky_golay(y, window_size, poly_order, deriv=1, rate=1/dt)
    
    # Combine to get velocity magnitude
    # We access the .y field to get the vector data
    vx = sg_x.y
    vy = sg_y.y
    
    v = sqrt.(vx.^2 .+ vy.^2)
    return v
end

function find_med(x,y,timeout_smoother,dt)
    timeout_smoother_steps = Int(ceil(timeout_smoother/dt))
    
    v = get_sg_velocity(x, y, timeout_smoother_steps, dt)

    # Threshold: λ=6 seems to be common for Engbert-Kliegl. 
    msd = median(abs.(v .- median(v))) / 0.6745
    return msd 
end

function calculate_isi(x, y, msd; timeout_smoother=0.007, timeout=0.150, λ=6.0, ratio=0.1, dt=0.0001, min_dur_steps=5)
    timeout_steps = Int(ceil(timeout/dt))
    timeout_smoother_steps = Int(ceil(timeout_smoother/dt))
    
    v = get_sg_velocity(x, y, timeout_smoother_steps, 0.0001)

    # Threshold: λ=6 is standard for Engbert-Kliegl. 
    threshold_on = λ * msd

    λ_off = ratio * λ
    threshold_off = λ_off * msd

    starts_timed = Int[]
    last_start = -timeout_steps 
    
    i = 1
    n = length(v)
    
    while i <= n
        # Wait for velocity to spike above the high onset threshold
        if v[i] > threshold_on
            j = i + 1
            
            # Keep moving forward until the particle is "effectively stationary", as defined by threshold_off
            while j <= n && v[j] >= threshold_off
                j += 1
            end
            
            event_duration = j - i
            
            # Check duration and Refractory period
            if event_duration >= min_dur_steps && (i - last_start) > timeout_steps
                push!(starts_timed, i)
                last_start = i
            end
            
            # Advance our search index to the end of the stationary tail
            i = j
        else
            i += 1
        end
    end

    isi = diff(starts_timed) .* (dt * 1000) 
    return (starts_timed, isi) 
end

isi_vec = Float64[]

### loop through csv files in /datadir

#Set accumulators
global μ::Float64 = 0
total_samples = length(csv_files)

for file in csv_files
    df = CSV.read(file, DataFrame)
    x = df.value1
    y = df.value2
    dt = 0.0001
    global μ += find_med(x,y,2,dt)/total_samples
end

for file in csv_files
    df = CSV.read(file, DataFrame)
    x = df.value1
    y = df.value2
    dt = 0.0001

    (sac_starts, isi) = calculate_isi(x, y, μ; 
        timeout_smoother=4, 
        timeout=0.05,
        ratio=0.1,
        λ=3,
        dt=dt, 
        min_dur_steps=1000
    )

    append!(isi_vec,isi)

    if length(isi) > 5 # Only plot if we have decent statistics

        p4 = plot(0.023.*(1:length(x)),x, label="X Position", color=:blue, alpha=0.6, legendfontsize = 12)
        plot!(0.023.*(1:length(y)),y, label="Y Position", color=:green, alpha=0.6)
        
        # Add scatter points on X trace to mark starts
        plot!(sac_starts*0.023, x[sac_starts], seriestype=:scatter, 
        markersize=10, color=:red, label="Detected Saccades")
        
        plot!(xlims=(200000*0.023, 300000*0.023), ylims=(-15,30), title="Saccade Detection Trace")
        
        outname4 = joinpath(output_dir, splitext(basename(file))[1] * "-SACCADES.pdf")
        savefig(p4, outname4)
        
    end
    
    println("Processed $FigNumber: $(basename(file))")
    global FigNumber += 1
end

# Rescaling to match Fries' data:
isi_vec .*= 0.023

N = length(isi_vec)
timelength = 1000
plotting_x = range(0,timelength)

# timelength over which we plot:
# # Precomputing the log normal fit
# lISI = log.(isi_vec)
# log_σ = std(lISI)
# RI = 1 / sqrt(exp(log_σ^2) - 1)
# log_peak = median(lISI)
# fitted_ln = LogNormal(log_peak,log_σ)

CV_2 = Float64[]
for i in firstindex(isi_vec):lastindex(isi_vec)-1
    cv2value = 2 * abs(isi_vec[i+1]-isi_vec[i]) / (isi_vec[i+1]-isi_vec[i])
    push!(CV_2, cv2value)
end

RCRI = 1/(1 + mean(CV_2))

# Tenth order Gamma model:
order = 10.
scale = mean(isi_vec) / order

Gamma_fit = Gamma(order, scale)

general_Gamma_fit = fit(Normal, isi_vec)

### Histogram with Gamma fits
p1 = histogram(isi_vec,
    bins = range(0, timelength, length=60),
    normalize = :pdf,
    xlabel = "Inter-Saccadic Interval (ms)",
    ylabel = "Probability Density",
    xlims = (0, timelength),
    label = "ISIs (RCRI=$(round(RCRI, digits=3)))", 
    color = :steelblue,           
    linecolor = :white,           
    linewidth = 0.5,
    fillalpha = 0.7
)

# # Overlay the 10th-order Gamma fit
# plot!(p1, plotting_x, pdf.(Gamma_fit, plotting_x), 
#     label = "Gamma(10)",
#     color = :darkred,          # Deep red for the fixed order model
#     linewidth = 2.5
# )

# # Overlay the General MLE Gamma fit
# plot!(p1, plotting_x, pdf.(general_Gamma_fit, plotting_x), 
#     label = "General MLE Gamma", 
#     color = :darkorange,       # Orange for the general model
#     linewidth = 2.5,
#     linestyle = :dash          # Dashed line to easily distinguish the two fits
# )

outname1 = joinpath(output_dir, "Total_ISIs.pdf")
savefig(p1, outname1)

# ### Histogram with Log-normal fit
# p1 = histogram(isi_vec,
#     bins = range(0, timelength, length=120),
#     normalize = :pdf,
#     xlabel = "Inter-Saccadic Interval (ms)",
#     ylabel = "Probability Density",
#     xlims = (0, timelength),
#     label = "Simulated ISI Data", # Fixed: Changed 'legend' to 'label'
#     color = :steelblue,           # Professional muted blue
#     linecolor = :white,           # White borders separate the bins clearly
#     linewidth = 0.5,
#     fillalpha = 0.7
# )

# plot!(p1, plotting_x, pdf.(fitted_ln, plotting_x), 
#     label = "Log-Normal (RI = $(round(RI, digits=3)))", 
#     color = :darkorange,          # Contrasting complementary color
#     linewidth = 2.5
# )

# outname1 = joinpath(output_dir, "Total_ISIs.pdf")
# savefig(p1, outname1)

## ISI timeseries
p2 = plot(isi_vec, 
    xlabel = "Index", 
    ylabel = "ISI (ms)", 
    title = "ISI Timeseries",
    label = false,                # No legend needed for a single line
    color = :gray30,              # Softer than pure black
    linewidth = 1.0,
    linealpha = 0.7               # Transparency helps reveal data density
    )
    
outname2 = joinpath(output_dir, "ISI_TIMESERIES.pdf")
savefig(p2, outname2)
    
## Poincaré plot

xs = isi_vec[1:end-1]
ys = isi_vec[2:end]

diff_vec = ys .- xs       # Perpendicular changes
sum_vec = ys .+ xs        # Longitudinal changes

sd1 = std(diff_vec) / sqrt(2)
sd2 = std(sum_vec) / sqrt(2)

# println("Rhythm Stability Metrics:")
# println("SD1 (Short-term jitter): ", round(sd1, digits=2))
# println("SD2 (Long-term drift):   ", round(sd2, digits=2))
# println("Ratio (SD2/SD1):         ", round(sd2/sd1, digits=2))
# println("Log SD:                  ", round(log_σ, digits=3))

# # Find absolute min/max to ensure perfectly square axes for the identity line
ax_min = min(minimum(xs), minimum(ys)) * 0.95
ax_max = max(maximum(xs), maximum(ys)) * 1.05

p3 = scatter(xs, ys, 
    label = "Intervals", 
    alpha = 0.5, 
    markersize = 2.5,             # Reduced marker size for cleaner overlap
    markerstrokewidth = 0,
    color = :steelblue,
    aspect_ratio = :equal, 
    xlims = (ax_min, ax_max),     # Lock axes to be perfectly identical
    ylims = (ax_min, ax_max),
    title = "Poincaré Plot (Return Map)",
    xlabel = "Interval t (ms)",
    ylabel = "Interval t+1 (ms)",
    legend = :topleft             # Move legend away from the identity line
)

# Add Line of Identity
plot!(p3, [ax_min, ax_max], [ax_min, ax_max], 
    label = "Line of Identity (y=x)", 
    color = :firebrick,           # Distinct dark red
    linestyle = :dash,
    linewidth = 1.5
)

poinc_outname = joinpath(output_dir, "POINCARE.pdf")
savefig(p3, poinc_outname)

# --- 1. Save the primary data ---
df_data = DataFrame(ISI = isi_vec)
CSV.write(joinpath(output_dir, "ISI_Timeseries_Data.csv"), df_data)

# --- 2. Save the metrics ---
# # Pack all your single-value variables into a 1-row DataFrame
# df_metrics = DataFrame(
#     TimeLength = timelength,
#     LogPeak = log_peak,
#     LogSigma = log_σ,
#     RI = RI,
#     SD1 = sd1,
#     SD2 = sd2
# )
# CSV.write(joinpath(output_dir, "ISI_Metrics.csv"), df_metrics)

println("Saved ISI data and metrics successfully!")
