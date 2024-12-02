# install the tidyverse
library(tidyverse)
library(patchwork)
library(scales)
library(ggthemes)
library(egg)
library(extrafont)
library(dplyr)

# font_install('fontcm')
font_import()
loadfonts()

# read data
df_raw = read.csv('data.txt', sep=";")

# mutations
df_raw$layer = df_raw$layer + 1

ssl = "self-supervised"
stl = "SKR STL C12"
joint = "joint C12"
dj2s6c = "disjoint 2s C6"
dj2s12c = "disjoint 2s C12"
dj10s6c = "disjoint 10s C6"
dj10s12c = "disjoint 10s C12"

df <- df_raw %>%
  mutate(
    weights = case_when(
      weights == "pre-trained" ~ ssl,
      weights == "STL C12" ~ stl,
      weights == "MTL joint C12" ~ joint,
      weights == "MTL dj 2s C6" ~ dj2s6c,
      weights == "MTL dj 2s C12" ~ dj2s12c,
      weights == "MTL dj 10s C6" ~ dj10s6c,
      weights == "MTL dj 10s C12" ~ dj10s12c,
      TRUE ~ weights  # Keep other values unchanged
    )
  )


custom_legend_order <- c(
  ssl,
  stl,
  joint,
  dj2s6c,
  dj2s12c,
  dj10s6c,
  dj10s12c
)

# theme
style = theme_linedraw()

# x_axis
x_axis = scale_x_continuous(
  name='layer for SKR evaluation', 
  breaks=seq(0, 12, by = 1),
  minor_breaks =NULL
)

# y_axis
y_axis =  scale_y_continuous(limits = c(0, 1), name="EER", labels = label_percent(accuracy=1))
y_lim = coord_cartesian(ylim=c(0, 0.50))

# plot
plot = (
  ggplot(df)
  + aes(
    layer, 
    EER, 
    color=weights,
    shape=weights
  )
  + geom_point()
  + geom_line()
  + scale_colour_colorblind(
    name='model',
    breaks=custom_legend_order,
    guide=guide_legend(nrow =8)
  )
  + scale_shape_manual(
    name='model',
    breaks=custom_legend_order,
    values=seq(0,6)
  )  
  + x_axis 
  + y_axis + y_lim 
  # + ggtitle('Layer-wise evaluation on vox2-dev')
  + style
  & theme(
    #legend.direction = "horizontal",
    #egend.position = "right",
    text         = element_text(family="mono"),
    axis.title.x = element_text(family="CM Roman"),
    axis.title.y = element_text(family= "CM Roman")
  )
)

plot

ggsave(
  file="layer_vs_eer.pdf",
  #device=cairo_pdf,
  width = 130,
  height = 90,
  units = "mm"
)
embed_fonts("layer_vs_eer.pdf", outfile="layer_vs_eer.pdf")
