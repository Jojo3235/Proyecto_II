library(ggplot2)
library(dplyr)

# Cargar el CSV
fuerzabruta <- read.csv2("fuerzabruta.csv")

# Renombrar las columnas según el CSV
names(fuerzabruta) <- c("name", "user", "comment")

ap_top_terms <- fuerzabruta %>%
  group_by(comment) %>%
  count(user, sort = TRUE) %>%
  group_by(comment) %>%
  slice_max(n = 10, order_by = n) %>%
  ungroup() %>%
  arrange(comment, -n)


# Gráfico
ap_top_terms %>%
  ggplot(aes(n, reorder(comment, n), fill = factor(comment))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ comment, scales = "free") +
  labs(y = "comment") +
  theme(axis.text.y = element_text(hjust = 0.5, vjust = 0.5, margin = margin(0, 4, 0, 4)))  # Ajuste de etiquetas de eje y
