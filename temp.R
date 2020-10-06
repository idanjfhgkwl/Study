library(tidyverse)

ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color = class)) +
  geom_smooth(se = FALSE) +
  labs(
    title = "엔진 크기가 증가할수록 일반적으로 연비는 감소함",
    subtitle = "2인승 차(스포츠카)는 중량이 작아서 예외",
    caption = "출처 fueleconomy.gov"
  )
