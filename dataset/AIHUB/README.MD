## AIHUB 데이터셋 사용

1. [Download](https://aihub.or.kr/aidata/33985)에서 Validation 만을 다운로드 합니다. 전체 데이터는 양이 너무 많아, 잘 엄선되어서 구성된 Validation만으로도 충분한 품질이 보장될 거라 생각합니다.
2. local terminal에서 scp를 통해 데이터를 서버로 올려줍니다. #21 을 참고해주세요 이 때 서버로 올리는 데이터는 [원천]Validation_간판1.zip , [원천]Validation_책표지.zip, [라벨]Validation.zip 세 개 뿐입니다. 간판 1,2,3 중에 1을 선택한 이유는 데이터셋에 가로간판이 제일 많아 보였기 때문입니다.( 실제 현실에서도 제일 많을 것 같았고 굳이 지금 당장 세분화된 데이터셋을 함께 넣을 필요는 없어 보였습니다.)
3. 데이터가 서버에 모두 올라갔다면 ( 약 1시간 소요 ) 각각의 파일을 unzip해줍니다. 그러면 현재 구성된 파일 디렉토리는 꽤나 복잡할 것입니다. 이를 저희가 정한 포맷에 맞게 재 구성해줘야합니다.

- [ ] unzip된 폴더 내에 같은 전체 데이터에 대한 zip파일이 하나씩 있습니다. 이를 지워줘야 합니다. ( shell script를 통해 for문으로 rm을 사용해서 지우셔도 됩니다. 다만, 이 경우는 하나하나 그냥 지우는 거나 비슷합니다.)
![image](https://user-images.githubusercontent.com/90104418/163917733-d4c4cf58-55f3-4dc5-92f7-d6648164f95b.png)

이제 구성된 데이터들을 저희가 원하는 포맷에 맞게 변환해야합니다. 이 때 간판 데이터인지 책표지 데이터인지에 따라서 달라집니다. 

- [ ] 간판 데이터의 경우 그냥 폴더 자체를 원하는 경로에 넣어두고, gt의 경우 폴더명만 gt로 image의 경우 폴더명만 image로 바꿔주면 됩니다. 
- [ ] 하지만, 책표지 데이터의 경우 각각의 폴더마다 데이터가 따로 나뉘어져 있기에, 이를 하나의 폴더 안으로 옮기기 위해서 mv커맨드를 사용해야합니다. 또한, 폴더 종류가 11개나 되기에 간단히 shell script를 사용하는 것이 좋습니다. shell scrript는 code의 dataset 폴더 안에 들어가면 확인할 수 있습니다. 해당 shell script를 images에 맞게 한 번 gt들에 맞게 한 번씩 조금씩 txt와 sh를 재구성해서 사용해주시면 됩니다. ( 모든 폴더 경로들을 얻기 위해서는 간단히 test.py를 통해 glob으로 얻는 것이 좋은 것 같습니다) 

이렇게 각각 gt와 images안에 넣어주게 되면 아래와 같이 나오게 될 것입니다.
![image](https://user-images.githubusercontent.com/90104418/163918299-5b31a3ca-e18b-4820-9461-ca017821690c.png)
각각의 gt와 image안에는 json과 image들이 쭉 나열되어 있을 것입니다. 이 형태가 맞춰주셔야 ufo포맷으로 변경이 깔끔해집니다.
![image](https://user-images.githubusercontent.com/90104418/163918414-88e4cb3a-b203-4913-8164-2a745c7cdd70.png)

- [ ] 그럼 이제 aihub_to_ufo.py에 argument로 경로만 잘 설정하셔서 넣어주시면 됩니다.

**자세한 변경 사항은 pr을 참고해주세요!**