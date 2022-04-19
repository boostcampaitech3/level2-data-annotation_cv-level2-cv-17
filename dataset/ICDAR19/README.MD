## ICDAR 19 다운과 포맷 변환

ICDAR 19의 경우 간단합니다. code의 dataset의 download.sh를 실행시켜주시면 ICDAR 데이터셋이 다운로드 됩니다.
이 때 images가 ImagesPart1,2로 나뉘어져 있기에 mv커맨드를 통해 해당 경로의 파일들을 모두 ICDAR19/images로 옮겨주시면 됩니다. 간단한 커맨드 예시는 mv [~/ImagesPart1경로/*] [옮기고 싶은 목적 경로 - ICDAR19/images] 입니다. 

이렇게 폴더 형식을 맞춘 ICDAR 19는 아래와 같은 형식을 가져야 합니다
![image](https://user-images.githubusercontent.com/90104418/163918999-0bffe758-4937-4321-aa18-656ba957152d.png)

후에는 convert_mlt.py를 실행해 경로 설정과 version을 19로만 설정해주시면 동작하게 됩니다.