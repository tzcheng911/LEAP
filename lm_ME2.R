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
alldata = read.csv("SSEP_sensor.csv")

## Mutate new factors and re-level
alldata = alldata %>%
  mutate(age = as.factor(age))
alldata = alldata %>%
  mutate(condition = as.factor(condition))
alldata = alldata %>%
  mutate(sub_id = as.factor(sub_id))
alldata$age = relevel(alldata$age,ref="7mo")
alldata$condition = relevel(alldata$condition,ref="random")

## Descriptive stats
summary_alldata = alldata %>%
  group_by(age,condition) %>%
  summarize(mMeterD = mean(X1.67Hz), mMeterT = mean(X1.11Hz), mBeat = mean(X3.3Hz),Nsubs=n_distinct(sub_id), 
            seMeterD = sd(X1.67Hz)/sqrt(Nsubs),seMeterT = sd(X1.11Hz)/sqrt(Nsubs),seBeat = sd(X3.3Hz)/sqrt(Nsubs))

## 2-way anova attempt 
duple_random = filter(alldata,condition != "triple")
summary(aov(X1.67Hz ~ age*condition+Error(sub_id/condition),data=duple_random)) 
triple_random = filter(alldata,condition != "duple")
summary(aov(X1.11Hz ~ age*condition+Error(sub_id/condition),data=triple_random)) 
summary(aov(X3.3Hz ~ age*condition+Error(sub_id/condition),data=alldata)) 

## one way lm across age
duple = filter(alldata,condition == "duple")
triple = filter(alldata,condition == "triple")
random = filter(alldata,condition == "random")

# meters
duple = duple %>%
  mutate(diff = duple$X1.67Hz-random$X1.67Hz)
triple = triple %>%
  mutate(diff = triple$X1.11Hz-random$X1.11Hz)
lmall = lmer(diff ~ age  + (1|sub_id),data= duple,verbose=2)  
summary(lmall) 
lmall = lmer(diff ~ age  + (1|sub_id),data= triple,verbose=2)  
summary(lmall) 

## one way anova 
m = aov(diff ~ age,data=duple)
summary(m) 
TukeyHSD(m)

m = aov(diff ~ age,data=triple)
summary(m) 
TukeyHSD(m)

# beat
duple = duple %>%
  mutate(diff = duple$X3.3Hz-random$X3.3Hz)
triple = triple %>%
  mutate(diff = triple$X3.3Hz-random$X3.3Hz)
lmall = lmer(diff ~ age  + (1|sub_id),data= duple,verbose=2)  
summary(lmall) 
lmall = lmer(diff ~ age  + (1|sub_id),data= triple,verbose=2)  
summary(lmall) 

## one way anova 
m = aov(diff ~ age,data=duple)
summary(m) 
TukeyHSD(m)

m = aov(diff ~ age,data=triple)
summary(m) 
TukeyHSD(m)

## duple lm
lmall = lmer(X1.67Hz ~ age*condition  + (1|sub_id),data= alldata,verbose=2,subset = condition != "triple")  
summary(lmall) 
lmall_noAge = lmer(X1.67Hz ~ age*condition-age:condition + (1|sub_id),data= alldata,verbose=2,subset = condition != "triple")  
anova(lmall,lmall_noAge)

## triple lm
lmall = lmer(X1.11Hz ~ age*condition  + (1|sub_id),data= alldata,verbose=2,subset = condition != "duple")  
summary(lmall) 
lmall_noAge = lmer(X1.11Hz ~ age*condition-age:condition + (1|sub_id),data= alldata,verbose=2,subset = condition != "duple")  
anova(lmall,lmall_noAge)

## beat lm
lmall = lmer(X3.3Hz ~ age*condition  + (1 |sub_id),data= alldata,verbose=2)  
summary(lmall) # Use early as the reference
lmall_noAge = lmer(X3.3Hz ~ age*condition-age:condition + (1|sub_id),data= alldata,verbose=2)  
anova(lmall,lmall_noAge)

## Visualization
ggplot(duple_random, aes(x = age, y = X1.67Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(triple_random, aes(x = age, y = X1.11Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(alldata, aes(x = age, y = X3.3Hz, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()
