# install.packages(c("readr","dplyr","ggplot2"))  # <- if needed

library(readr)
library(dplyr)
library(ggplot2)

# ---- CONFIG ----
csv_path <- "data/outputs/mosquito_outputs_0813/all_images_long.csv"  # <- adjust as needed

out_grouped_pdf <- "plots/boxplot_0813.pdf"

# ---- LOAD ----
df <- read_csv(csv_path, show_col_types = FALSE) %>%
  mutate(
    experiment_id = as.factor(experiment_id),
    timepoint_min = as.integer(timepoint_min),
    mosquito_count = as.numeric(mosquito_count)
  ) %>%
  filter(!is.na(timepoint_min), !is.na(mosquito_count))

# Ordered x-axis (one box per (experiment_id, timepoint))
time_levels <- sort(unique(df$timepoint_min))
df <- df %>% mutate(timepoint_min_f = factor(timepoint_min, levels = time_levels))

# Medians per timepoint & experiment (for trend lines)
med_df <- df %>%
  group_by(experiment_id, timepoint_min, timepoint_min_f) %>%
  summarise(median_count = median(mosquito_count, na.rm = TRUE), .groups = "drop")

# ---- PLOT: Grouped boxplots on top of each other + trend lines ----
p_grouped <-
  ggplot(df, aes(x = timepoint_min_f, y = mosquito_count, fill = experiment_id)) +
  geom_boxplot(
    aes(group = interaction(timepoint_min_f, experiment_id),
        fill  = experiment_id,
        color = experiment_id),
    position = position_identity(),
    width = 0.7,
    alpha = 0.35,        # transparency so overlaps are visible
    outlier.alpha = 0.4
  ) +
  # trend lines of medians across time (one line per experiment)
  geom_line(
    data = med_df,
    aes(x = timepoint_min_f, y = median_count, color = experiment_id, group = experiment_id),
    linewidth = 0.9
  ) +
  geom_point(
    data = med_df,
    aes(x = timepoint_min_f, y = median_count, color = experiment_id),
    size = 2
  ) +
  labs(
    title = "Szúnyog szám idő viszonylatában",
    x = "idő (perc)",
    y = "Szúnyog szám (db)",
    fill = "kísérleti felállás",
    color = "kísérleti felállás"
  ) +
  scale_x_discrete(labels = as.character(time_levels)) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(out_grouped_pdf, p_grouped, width = 11, height = 6.5, dpi = 150)

cat("Saved:\n",
    "-", out_grouped_pdf, "\n")

