library(lme4)
library(lmerTest)
library(tidyverse)
library(dplyr)
library(broom)
library(ggplot2)
library(ggpubr)
library('fastDummies')
library(effsize)
library(lsr)
library(rstatix)
library(car)

## Load the data
alldata = read.csv("/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/SSEP/SSEP_sensor.csv")

## Mutate new factors and re-level
alldata = alldata %>%
  mutate(fage = as.factor(age))
alldata = alldata %>%
  mutate(fcondition = as.factor(condition))
alldata$fage = relevel(alldata$fage,ref="7mo")
alldata$fcondition = relevel(alldata$fcondition,ref="random")

## Visualization
ggplot(alldata, aes(x = fage, y = X1.11Hz, fill = fcondition)) +
  geom_bar(stat="summary", fun.y = "mean", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(alldata, aes(x = fage, y = X1.67Hz, fill = fcondition)) +
  geom_bar(stat="summary", fun.y = "mean", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(alldata, aes(x = fage, y = X3.3Hz, fill = fcondition)) +
  geom_bar(stat="summary", fun.y = "mean", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

## lm
lmall = lmer(X1.11Hz ~ fage*fcondition  + (1 + fcondition|sub_id),data= alldata,control = glmerControl(optimizer="bobyqa"), verbose=2)  
summary(lmall) # Use early as the reference