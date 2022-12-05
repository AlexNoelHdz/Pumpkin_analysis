library(tidyverse)
library(broom)
library(gt)
library(patchwork)
library(tictoc)
library(blorr)
library(readxl)

#Poner semilla
set.seed(312)
#Archivo a utilizar
Pumpkin_Seeds_Dataset <- read_excel("C:/Users/L00739961/Desktop/Respaldo febrero 15 2022/Iteso/Optimizaci�n Convexa/Proyecto/Archivos/Pumpkin_Seeds_Dataset.xlsx")

#Renombrar el archivo

default<-Pumpkin_Seeds_Dataset

glimpse(default)

#tomar toda la muestra para graficar

d<-default %>%
  add_count(Class,name="n_group") %>%
  slice_sample(
    n=2500,
    weight_by=n()-n_group
  )

# 1 Combinaci�n de variables para ver la clasificaci�n
#con los coeficientes de menor valor absoluto.

p1<-d %>%
  ggplot(aes(x=Area, y=Perimeter))+
  geom_point(aes(color=Class,shape=Class),
             alpha=0.5, show.legend=FALSE)
p2<-d %>%
  ggplot(aes(x=Area,y=Perimeter))+
  geom_boxplot(aes(fill=Class), show.legend=FALSE)
p3<-d %>%
  ggplot(aes(x=Area, y=Perimeter))+
  geom_boxplot(aes(fill=Class),show.legend=FALSE)
p1|(p2|p3)

# 2 Nueva combinaci�n de variables

p4<-d %>%
  ggplot(aes(x=Area, y=Major_Axis_Length))+
  geom_point(aes(color=Class,shape=Class),
             alpha=0.5, show.legend=FALSE)
p5<-d %>%
  ggplot(aes(x=Area,y=Major_Axis_Length))+
  geom_boxplot(aes(fill=Class), show.legend=FALSE)
p6<-d %>%
  ggplot(aes(x=Area, y=Major_Axis_Length))+
  geom_boxplot(aes(fill=Class),show.legend=FALSE)
p4|(p5|p6)

# 3 Nueva combinaci�n de variables

p7<-d %>%
  ggplot(aes(x=Extent, y=Aspect_Ration))+
  geom_point(aes(color=Class,shape=Class),
             alpha=0.5, show.legend=FALSE)
p8<-d %>%
  ggplot(aes(x=Extent,y=Aspect_Ration))+
  geom_boxplot(aes(fill=Class), show.legend=FALSE)
p9<-d %>%
  ggplot(aes(x=Extent, y=Aspect_Ration))+
  geom_boxplot(aes(fill=Class),show.legend=FALSE)
p7|(p8|p9)

# 4 Nueva combinaci�n de variables

p10<-d %>%
  ggplot(aes(x=Perimeter, y=Aspect_Ration))+
  geom_point(aes(color=Class,shape=Class),
             alpha=0.5, show.legend=FALSE)
p11<-d %>%
  ggplot(aes(x=Perimeter,y=Aspect_Ration))+
  geom_boxplot(aes(fill=Class), show.legend=FALSE)
p12<-d %>%
  ggplot(aes(x=Perimeter, y=Aspect_Ration))+
  geom_boxplot(aes(fill=Class),show.legend=FALSE)
p10|(p11|p12)

# 5 Nueva combinaci�n de variables con las de mayor valor absoluto.

p13<-d %>%
  ggplot(aes(x=Solidity, y=Aspect_Ration))+
  geom_point(aes(color=Class,shape=Class),
             alpha=0.5, show.legend=FALSE)

p14<-d %>%
  ggplot(aes(x=Solidity,y=Aspect_Ration))+
  geom_boxplot(aes(fill=Class), show.legend=FALSE)

p15<-d %>%
  ggplot(aes(x=Solidity, y=Aspect_Ration))+
  geom_boxplot(aes(fill=Class),show.legend=FALSE)

p13|(p14|p15)


#Regresion Log�stica

df<-default %>% select(Class,Area,Perimeter,Major_Axis_Length,Minor_Axis_Length,Solidity,Extent,Aspect_Ration )
sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.7,0.3))
train  <- df[sample, ]
test   <- df[!sample, ]

glm.fit<-
  glm(
    Class~Area+Perimeter+Major_Axis_Length+Minor_Axis_Length+Solidity+Extent+Aspect_Ration,
    data=train %>% mutate(Class=ifelse(Class=='�er�evelik',1,0))
  )

#es una paqueter�a que da informaci�n del modelo

blorr::blr_model_fit_stats(glm.fit)

blr_regress(glm.fit)

glm.probs <- predict(glm.fit,
                     newdata = test,
                     type = "response")

glm.pred <- ifelse(glm.probs > 0.5, "�er�evelik", "�rg�p Sivrisi")

#matriz de confusi�n, da los falsos positivos, se hace con las
#predicciones del test y los datos reales

confusionmatrix<-table(glm.pred,test$Class)

#C�lculo de la precisi�n 0.624/697
#Accuracy

mean(glm.pred==test$Class)

###test solo para accuracy
