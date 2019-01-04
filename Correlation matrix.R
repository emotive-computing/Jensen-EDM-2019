library(corrplot)

scaled_data <- read.csv('Processed Usage Features.csv',row.names = 1)
scaled_data <- read.csv('All state predictions.csv')
scaled_data$student_id <- NULL

res <- cor(scaled_data, method = "spearman")
res <- round(res,2)

col <- colorRampPalette(c("blue4","white","darkred"))(20)

corrplot(res, method = "number", col = c("black"), cl.pos = "n",type = "upper",
         tl.col = "black", tl.srt = 45)
corrplot(res, method = "color", addCoef.col="white",number.cex=0.75, 
         type = "upper", tl.srt = 45, tl.col = "black")