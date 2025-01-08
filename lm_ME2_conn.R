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

################################################################# Load the ROI data
alldata = read.csv("connectivity_roi.csv")

# log transform to achieve normality
alldata = alldata %>% 
  mutate(log_Delta.conn = log(Delta.conn),log_Theta.conn = log(Theta.conn), log_Alpha.conn = log(Alpha.conn), log_Beta.conn = log(Beta.conn))

# Mutate new factors and re-level
alldata = alldata %>%
  mutate(age = as.factor(age))
alldata = alldata %>%
  mutate(condition = as.factor(condition))
alldata = alldata %>%
  mutate(sub_id = as.factor(sub_id))

alldata$age = relevel(alldata$age,ref="7mo")
alldata$condition = relevel(alldata$condition,ref="_02")

## Descriptive stats
# Connectivity
summary_alldata = alldata %>%
  group_by(age,condition) %>%
  summarize(mDeltaConn = mean(Delta.conn), mThetaConn = mean(Theta.conn), mAlphaConn = mean(Alpha.conn), mBetaConn = mean(Beta.conn),Nsubs=n_distinct(sub_id), 
            seDeltaConn = sd(Delta.conn)/sqrt(Nsubs),seThetaConn = sd(Theta.conn)/sqrt(Nsubs),seAlphaConn = sd(Alpha.conn)/sqrt(Nsubs),seBetaConn = sd(Beta.conn)/sqrt(Nsubs))

## 2-way Mixed effect ANOVA 
# check assumptions - outliers
outliers = alldata %>%
  group_by(condition,age) %>%
  identify_outliers(log_Alpha.conn) # change to Delta.conn, X1.67Hz and Alpha.conn
alldata = filter(alldata, !(sub_id %in% unique(outliers$sub_id)))
alldata = filter(alldata, !(sub_id %in% 'br_13')) # exclude br_13

# check assumptions - Normality (p > 0.05) log_X3.3Hz failed
alldata %>%
  group_by(condition,age) %>%
  shapiro_test(log_Alpha.conn) # change to Delta.conn, X1.67Hz and X3.3Hz

# check assumptions - Homogneity of variance (p > 0.05) log_X1.67Hz failed
alldata %>%
  group_by(condition) %>%
  levene_test(log_Alpha.conn ~ age) # change to Delta.conn, X1.67Hz and X3.3Hz

# check assumptions - Homogeneity of covariances (unless p < 0.001 and your sample sizes are unequal, ignore it)
box_m(alldata[,"log_Alpha.conn",drop=FALSE],alldata$age)

################################################################# run ANOVA
duple_random = filter(alldata,condition != "_04")

res.aov <- anova_test(
  data = duple_random, dv = log_Delta.conn, wid = sub_id,
  between = age, within = condition
)

get_anova_table(res.aov)

res.aov <- anova_test(
  data = duple_random, dv = log_Theta.conn, wid = sub_id,
  between = age, within = condition
)
get_anova_table(res.aov)

res.aov <- anova_test(
  data = duple_random, dv = log_Alpha.conn, wid = sub_id,
  between = age, within = condition
)

get_anova_table(res.aov)

res.aov <- anova_test(
  data = duple_random, dv = log_Beta.conn, wid = sub_id,
  between = age, within = condition
)
get_anova_table(res.aov)

triple_random = filter(alldata,condition != "_03")
res.aov <- anova_test(
  data = triple_random, dv = log_Delta.conn, wid = sub_id,
  between = age, within = condition
)

get_anova_table(res.aov)

res.aov <- anova_test(
  data = triple_random, dv = log_Theta.conn, wid = sub_id,
  between = age, within = condition
)
get_anova_table(res.aov)

res.aov <- anova_test(
  data = triple_random, dv = log_Alpha.conn, wid = sub_id,
  between = age, within = condition
)

get_anova_table(res.aov)

res.aov <- anova_test(
  data = triple_random, dv = log_Beta.conn, wid = sub_id,
  between = age, within = condition
)
get_anova_table(res.aov)

# simple main effect of age
one.way <- duple_random %>%
  group_by(condition) %>%
  anova_test(dv = log_Delta.conn, wid = sub_id, between = age) %>%
  get_anova_table() %>%
  adjust_pvalue(method = "bonferroni")
one.way

## pairwise comparison of age and condition
# condition effect of each of the three ages
pwc <- duple_random %>%
  group_by(age) %>%
  pairwise_t_test(log_Delta.conn ~ condition, p.adjust.method = "bonferroni")
pwc

pwc <- duple_random %>%
  group_by(age) %>%
  pairwise_t_test(log_Theta.conn ~ condition, p.adjust.method = "bonferroni")
pwc

pwc <- duple_random %>%
  group_by(age) %>%
  pairwise_t_test(log_Alpha.conn ~ condition, p.adjust.method = "bonferroni")
pwc

pwc <- duple_random %>%
  group_by(age) %>%
  pairwise_t_test(log_Beta.conn ~ condition, p.adjust.method = "bonferroni")
pwc

# age effect of each of the two conditions
pwc <- duple_random %>%
  group_by(condition) %>%
  pairwise_t_test(log_Delta.conn ~ age, p.adjust.method = "bonferroni")
pwc

## Visualization
ggplot(duple_random, aes(x = age, y = Delta.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(triple_random, aes(x = age, y = Theta.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(alldata, aes(x = age, y = Alpha.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

ggplot(alldata, aes(x = age, y = Beta.conn, fill = condition)) +
  geom_bar(stat="summary", position='dodge') +
  stat_summary(fun.data=mean_se, geom="errorbar", position = position_dodge(width = 0.9), width=.1,color="grey") +
  geom_point(position = position_jitterdodge(jitter.width = 0.3,dodge.width = 0.9), color="black")+
  theme_bw()

## 2-way Mixed effect lmer
# duple lmer
lmall = lmer(X1.67Hz ~ age*condition  + (1|sub_id),data= alldata,verbose=2,subset = condition != "_04")  
summary(lmall) 
lmall_noAge = lmer(X1.67Hz ~ age*condition-age:condition + (1|sub_id),data= alldata,verbose=2,subset = condition != "triple")  
anova(lmall,lmall_noAge)

# triple lmer
lmall = lmer(X1.11Hz ~ age*condition  + (1|sub_id),data= alldata,verbose=2,subset = condition != "_03")  
summary(lmall) 
lmall_noAge = lmer(X1.11Hz ~ age*condition-age:condition + (1|sub_id),data= alldata,verbose=2,subset = condition != "duple")  
anova(lmall,lmall_noAge)

## beat lmer
lmall = lmer(X3.3Hz ~ age*condition  + (1 |sub_id),data= alldata,verbose=2)  
summary(lmall) # Use early as the reference
lmall_noAge = lmer(X3.3Hz ~ age*condition-age:condition + (1|sub_id),data= alldata,verbose=2)  
anova(lmall,lmall_noAge)
