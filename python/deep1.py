import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import gpuTool
from sendGpuInfo import updateGpuInfo
import multiprocessing
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


# IMDb 데이터셋 클래스 정의
class IMDbDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def getPowerDrawPerSecond(shared_isStart, shared_powerDraw_1epoch):
    sumPowerDraw = 0

    while True:
        if shared_isStart:
            sumPowerDraw = gpuTool.getGpuPowerDraw()
            shared_powerDraw_1epoch.value += float(sumPowerDraw)
            time.sleep(1)


def train_bert_on_imdb(shared_isStart, shared_powerDraw_1epoch):
    # 데이터셋 불러오기 및 전처리
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_dataset("imdb")
    train_data = IMDbDataset(dataset['train'], tokenizer)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    # BERT 모델 로드
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    max_Clock = 1950
    prevPerformance = None
    prevPowerDraw = None
    isUp = False
    isClockDone = False

    sumPowerDraw = 0
    sumPerformance = 0

    for epoch in range(num_epochs):
        start = time.time()  # 성능 측정 시작
        shared_isStart = True  # 파워소모 측정 시작

        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', total=len(train_loader), leave=False)

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()

            # backward pass
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=epoch_loss / (progress_bar.n + 1))  # 평균 손실 업데이트

        end = time.time()
        shared_isStart = False  # 파워소모 측정 종료
        performance = end - start

        # 성능 누적
        sumPerformance += performance

        # 여기서부터 156번줄까지 주석하면 default로로
       # default가 최고성능, 첫번째 에폭때엔 클럭 -30%
       #  if epoch == 0:
       #      print(max_Clock * 0.8)
       #      best_performance = performance
       #      gpuTool.setGpuClock(int(max_Clock * 0.8))
       #      isUp = False
       #  else:
       #      if ((abs(((performance - prevPerformance) / prevPerformance) * 100) < 2) and (
       #              abs(((performance - prevPerformance) / prevPerformance) * 100) > 0)) or \
       #              ((abs(((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100) < 2) and \
       #               (abs(((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100) > 0)):
       #          # 성능저하차이도 적고, 파워소모 차이도 적으면 그 클럭을 계속 유지
       #          print("클럭을 결정")
       #          isClockDone = True
       #      if not isClockDone:
       #          if not isUp:  # 클럭을 내렸을때
       #              print(f"성능차이{((performance - prevPerformance) / prevPerformance) * 100}")
       #              print(f"파워소모차이{((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100}")
       #              if (((performance - prevPerformance) / prevPerformance) * 100 > 10) or (
       #                      (shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100 > 2:
       #                  print(f"성능저하 {((performance - prevPerformance) / prevPerformance) * 100}")
       #                  gpuTool.setGpuClock(int(gpuTool.getGpuClock() * 1.1))  # 10퍼센트 올림
       #                  isUp = True
       #
       #
       #              else:
       #                  # 위 조건을 만족하지 않으면 클럭을 더 내리기
       #                  gpuTool.setGpuClock(int(gpuTool.getGpuClock() * 0.9))
       #
       #              # 클럭을 내렸을때 차이가 별로 없다면 이제 그 클럭으로 고정해도 되지않을까?
       #          elif isUp:  # 클럭을 올렸을때
       #              print(f"파워소모차이 {((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw)}")
       #              if ((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100 > 10:
       #                  print(f"파워소모량 증가 {((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100}")
       #                  gpuTool.setGpuClock(int(gpuTool.getGpuClock() * 0.9))
       #                  isUp = False
       #
       #              else:
       #                  # 위 조건을 만족하지 않으면 클럭을 더 올리기
       #                  gpuTool.setGpuClock(int(gpuTool.getGpuClock() * 1.1))
        #여기까지 주석
        sumPowerDraw += shared_powerDraw_1epoch.value

        prevPerformance = performance
        prevPowerDraw = shared_powerDraw_1epoch.value
        print(f"걸린 시간{performance}sec")
        print(f"소모 전력 (1에폭) {shared_powerDraw_1epoch.value}")

        shared_powerDraw_1epoch.value = 0  # 1에폭 소모량 초기화

        print(f"Epoch {epoch + 1} finished with loss: {epoch_loss / len(train_loader):.4f}")

    print("Training complete.")
    print(f"총 소모 전력 : {sumPowerDraw}")
    print(f"총 걸린 시간 : {sumPerformance}")


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_isStart = manager.Value('b', False)
    shared_powerDraw_1epoch = manager.Value('d', 0.0)

    #profiler_process = multiprocessing.Process(target=updateGpuInfo)
    profiler_process2 = multiprocessing.Process(target=getPowerDrawPerSecond,
                                                args=(shared_isStart, shared_powerDraw_1epoch))
    profiler_process3 = multiprocessing.Process(target=train_bert_on_imdb,
                                                args=(shared_isStart, shared_powerDraw_1epoch))

    #profiler_process.start()
    profiler_process2.start()
    profiler_process3.start()

    # train_bert_on_imdb 프로세스가 종료될 때까지 대기
    profiler_process3.join()

    # 나머지 프로세스를 종료
    #profiler_process.terminate()
    profiler_process2.terminate()

    # 프로세스가 종료될 때까지 기다림
    #profiler_process.join()
    profiler_process2.join()

    gpuTool.resetGpuClock()
