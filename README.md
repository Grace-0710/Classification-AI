# Classification-AI
Classification 관련 모델 

[ 딥러닝을 활용한 자연어처리 입문 ]

Part1. 딥러닝 기초
MNIST Classification 예제

* trainer.py
  - train 함수 참고자료
  
      <img width="315" alt="스크린샷 2023-03-17 오후 11 56 34" src="https://user-images.githubusercontent.com/84004919/225941220-6d95694b-e638-42be-a4ef-1065e6f83c90.png">
      
  - _train시 miniBatch로 쪼개기전에 데이터 shuffle => r=random하게 
  
    <img width="270" alt="스크린샷 2023-03-18 오전 12 00 33" src="https://user-images.githubusercontent.com/84004919/225942379-c5e4f8f5-ac2b-4ce2-a420-cd004b7b0eda.png">

  - index_select 역할 : input차원을 따라 텐서를 인덱싱하는 새 텐서를 반환
  
    <img width="320" alt="스크린샷 2023-03-18 오전 12 05 24" src="https://user-images.githubusercontent.com/84004919/225943505-2f7eef38-40ff-4936-8ca5-c0025a39e639.png">
    
  - zero_grad
    보통 딥러닝에서는 미니배치+루프 조합을 사용해서 parameter들을 업데이트하는데,
    한 루프에서 업데이트를 위해 loss.backward()를 호출하면 각 파라미터들의 .grad 값에 변화도가 저장이 된다.
    이후 다음 루프에서 zero_grad()를 하지않고 역전파를 시키면 이전 루프에서 .grad에 저장된 값이 다음 루프의 업데이트에도 간섭을 해서 원하는 방향으로 학습이 안된다고 한다.
    따라서 루프가 한번 돌고나서 역전파를 하기전에 반드시 zero_grad()로 .grad 값들을 0으로 초기화시킨 후 학습을 진행해야 한다.

* train.py : trainer를 train.py가 data loader와 함꼐 들고서 학습 진행 == 최상단 코드
  - main
  
    <img width="293" alt="스크린샷 2023-03-18 오전 12 25 39" src="https://user-images.githubusercontent.com/84004919/225948484-8d0e11e3-c5e0-4a96-a4a1-8984fe968a4d.png">

    
