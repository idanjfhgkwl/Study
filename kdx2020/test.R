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




