shinhancard

shinhancard2 <- shinhancard

colnames(shinhancard2) <- c('date', 'sex', 'age', 'business_type', 'case_number')

shinhancard2

ggplot (data = shinhancard) +
  geom_point(mapping = aes(x = 연령대별, y = `카드이용건수(천건)`))

ggplot(shinhancard2, aes(age, case_number)) +
  geom_point(mapping = aes(x = age, y = case_number)) +
  geom_smooth(se = FALSE) +
  labs(
    title = "성별, 연령대"
    )

ggplot(shinhancard2, aes(business_type, case_number)) +
  geom_col()

diamonds

ggplot(data = diamonds) +
  geom_count(mapping = aes(x = cut, y = color))

ggplot(data = shinhancard2) +
  geom_count(mapping = aes(x = age, y = sex))

products2 <- products

colnames(products2) <- c('id', 'date', 'category', 'sex', 'age', 'os', 'amount', 'purchase')

products2[2:8]

ggplot (data = products2[2:8]) +
  geom_point(mapping = aes(x = sex, y = age))

shinhancard2
s



mbnnews <- read_xlsx("data/mk_news_201901_202006.xlsx")
head(shinhancard)
shinhancard <- shinhancard %>% 
  select(-c(6:8))
head(shinhancard)






shinhancard - 카드이용건수
products - 구매수

# 일별 카드이용건수 by 카테고리

shinhancard %>% 
  rename(date = "일별") %>%  
  filter(업종 == "M001_한식") %>%
  group_by(date, 업종) %>% 
  summarise(mean = mean(`카드이용건수(천건)`)) %>%
  ggplot(aes(x=date)) + 
  geom_line(aes(y=mean, colour = 업종)) + 
  labs(title="Time Series Chart", 
       subtitle="Avg. Sales from 'Shinhan Card' Dataset", 
       caption="Source: Shinhan Card", 
       y="Avg. Sales (1000)")

# 업종별 구매건수(평균)

cat_rev <- shinhan_afco %>%
  group_by(업종) %>% 
  summarise(mean = mean(`카드이용건수(천건)`)) %>% 
  arrange(desc(mean)) 

shinhancard
cat_rev

cat_rev <- cat_rev[order(cat_rev$mean, decreasing = TRUE), ]  # sort
cat_rev$업종 <- factor(cat_rev$업종, levels = cat_rev$업종)  # to retain the order in plot.
head(cat_rev)
glimpse(cat_rev)
cat_rev
arrange(cat_rev, desc(업종))

head(shinhan_afco)

shinhan_afco %>%
  group_by(업종) %>% 
  summarise(mean = mean(카드이용건수.천건.)) %>% 
  arrange(업종) 

cat_rev <- shinhan_afco %>%
  group_by(업종) %>% 
  summarise(mean = mean(카드이용건수.천건.)) %>% 
  arrange(desc(mean)) 

cat_rev <- cat_rev[order(cat_rev$mean, decreasing = TRUE), ]  # sort
cat_rev$업종 <- factor(cat_rev$업종, levels = cat_rev$업종)  # to retain the order in plot.
head(cat_rev)

ggplot(cat_rev %>% head(20), aes(x = 업종, y = mean)) + 
  geom_bar(stat="identity", width=.5, fill="tomato3") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Ordered Bar Chart", 
       subtitle="업종 Vs Avg. 구매금액", 
       caption="source: products")

shinhan_afco %>% 
  rename(date = "년월") %>%  
  filter(업종 == "한식") %>%
  group_by(date, 업종) %>% 
  summarise(mean = mean(카드이용건수.천건.)) %>%
  ggplot(aes(x=date)) + 
  geom_line(aes(y=mean, colour = 업종)) + 
  labs(title="Time Series Chart", 
       subtitle="Avg. Sales from 'Shinhan Card' Dataset", 
       caption="Source: Shinhan Card", 
       y="Avg. Sales (1000)")

glimpse(shinhan_afco)

rm(samsung_card)

# 

install.packages("nycflights13")
library(nycflights13)
library(ggplot2)
nf <- count(flights, year, month, day)
library(lubridate)
nf <- mutate(nf, date = make_date(year, month, day))
library(ggplot2)
p <- ggplot(nf, aes(date, n)) + geom_line()
p
p + facet_wrap(~ wday(date, TRUE)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
monthweek <- function(d, w) ceiling((d - w) / 7) + 1
nf <- mutate(nf, wd = wday(date, label = TRUE))
nf <- mutate(nf, wd = factor(wd, levels = rev(levels(wd))))
nf <- mutate(nf, mw = monthweek(day, wday(date)))
ggplot(nf, aes(x = as.character(mw), y = wd, fill = n)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white") +
  facet_wrap(~ month(date, TRUE)) +
  ylab("") + xlab("Week of Month") +
  theme(panel.grid.major = element_blank())
nf2 <- mutate(nf, wd = wday(date, label = TRUE))
nf2 <- mutate(nf2, wd = factor(wd))
nf2 <- mutate(nf2, mw = factor(monthweek(day, wday(date))))
nf2 <- mutate(nf2, mw = factor(mw, rev(levels(mw))))
ggplot(nf2, aes(x = wd, y = mw, fill = n)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white") +
  facet_wrap(~ month(date, TRUE)) +
  ylab("") + xlab("Week of Month") +
  theme(panel.grid.major = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))
head(nf2)

head(shinhan_temp)
monthweek <- function(d, w) ceiling((d - w) / 7) + 1
shinhan_temp <- mutate(shinhan_covid, wd = wday(일별, label = TRUE))
shinhan_temp <- mutate(shinhan_temp, wd = factor(wd))
shinhan_temp <- mutate(shinhan_temp, mw = factor(monthweek(day, wday(일별))))
shinhan_temp <- mutate(shinhan_temp, mw = factor(mw, rev(levels(mw))))
nf <- count(flights, year, month, day)
glimpse(shinhan_temp)
shinhan_temp %>%
  group_by(일별, 업종번호, `카드이용건수(천건)`, 코로나, wd, mw) %>%
  filter(업종번호 == "M001") %>%
  ggplot(aes(x = wd, y = mw, fill = `카드이용건수(천건)`)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white") +
  facet_wrap(~ month(일별, TRUE)) +
  ylab("") + xlab("Week of Month") +
  theme(panel.grid.major = element_blank(),
        axis.text.x = element_text(angle = 0, hjust = 1))

shinhan_temp %>%
  select(일별, 업종번호, `카드이용건수(천건)`, 코로나, wd, mw)
  group_by(일별, 업종번호, `카드이용건수(천건)`, 코로나, wd, mw) %>%
  filter(업종번호 == "M001" & (코로나 == 2019 | 코로나 == 2020))


rm(monthweek)

rm(shinhan_temp)

#

install.packages("ggcharts")
library(dplyr)
library(ggplot2)
library(ggcharts)
install.packages("tidytext")
library(tidytext)
data("biomedicalrevenue")

biomedicalrevenue %>%
  filter(year %in% c(2012, 2015, 2018)) %>%
  bar_chart(x = company, y = revenue, facet = year, top_n = 10)

shinhan_covid_pv %>%
  filter(코로나 %in% c(2019, 2020)) %>%
  bar_chart(x = 업종번호, y = `카드이용건수(천건)`, facet = 코로나, top_n = 10)

head(biomedicalrevenue)
head(shinhan_covid)
head(shinhan_covid_pv)
glimpse(shinhan_covid_pv)

shinhan_covid_pv <- dcast(shinhan_covid, 코로나 ~ 업종번호, value.var="카드이용건수(천건)", mean)
rm(biomedicalrevenue)

#

data("revenue_wide")
library(ggcharts)
line_chart(data = shinhan_covid, x = year(일별), y = `카드이용건수(천건)`)

pyramid_chart(data = popch, x = age, y = pop, group = sex)
library(dplyr, warn.conflicts = FALSE)

pyramid_chart(
  data,
  x,
  y,
  group,
  bar_colors = c("#1F77B4", "#FF7F0E"),
  sort = "no",
  xlab = NULL,
  title = NULL
)
