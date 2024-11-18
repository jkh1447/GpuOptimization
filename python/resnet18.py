import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import gpuTool
from sendGpuInfo import updateGpuInfo
import multiprocessing

def getPowerDrawPerSecond(shared_isStart, shared_powerDraw_1epoch):
    sumPowerDraw = 0

    while True:
        if (shared_isStart):
            sumPowerDraw = gpuTool.getGpuPowerDraw()
            shared_powerDraw_1epoch.value += float(sumPowerDraw)
            time.sleep(1)

# 디바이스 설정 (GPU 사용 여부)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 입력 크기에 맞추어 MNIST 이미지를 리사이즈
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# ResNet18 모델 불러오기 (입력 채널을 1로 변경)
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        # Pretrained ResNet18 모델을 불러오고, 첫 번째 conv 레이어와 마지막 fc 레이어 수정
        self.model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 입력 채널을 1로 수정
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # 출력 클래스를 10으로 변경 (MNIST)

    def forward(self, x):
        return self.model(x)


# 모델, 손실 함수, 옵티마이저 정의
model = CustomResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# max clock
max_Clock = 1950

# 학습 함수
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 100번째 배치마다 진행 상황 출력
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch}, Batch: {batch_idx + 1}, Loss: {running_loss / 100:.6f}')
            running_loss = 0.0
    print(f'Epoch {epoch} 완료.')


# 학습 및 진행 상태만 표시

def resNet18(shared_isStart, shared_powerDraw_1epoch) :

    print("딥러닝 시작")
    epochs = 5

    best_performance = None
    performance = None
    prevPerformance = None
    prevPowerDraw = None
    isUp = False
    max_Clock = 1950

    isClockDone = False

    sumPowerDraw = 0
    sumPerformance = 0

    for epoch in range(1, epochs + 1):
        start = time.time()
        shared_isStart = True  # 파워소모측정 시작
        train(model, device, train_loader, optimizer, criterion, epoch)
        end = time.time()

        shared_isStart = False  # 파워소모 측정 종료
        performance = end - start

        sumPerformance += performance

        # default가 최고성능, 첫번째 에폭때엔 클럭 -30%
        # if epoch == 1:
        #     print(max_Clock * 0.8)
        #     best_performance = performance
        #     gpuTool.setGpuClock(int(max_Clock * 0.8))
        #     isUp = False
        # else:
        #     if ((abs(((performance - prevPerformance) / prevPerformance) * 100) < 2) and (abs(((performance - prevPerformance) / prevPerformance) * 100) > 0)) or \
        #             ((abs(((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100) < 2) and \
        #             (abs(((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100) > 0)):
        #         #성능저하차이도 적고, 파워소모 차이도 적으면 그 클럭을 계속 유지
        #         print("클럭을 결정")
        #         isClockDone = True
        #     if not isClockDone:
        #         if not isUp:  # 클럭을 내렸을때
        #             print(f"성능차이{((performance - prevPerformance) / prevPerformance) * 100}" )
        #             print(f"파워소모차이{((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100}")
        #             if (((performance - prevPerformance) / prevPerformance) * 100 > 10) or (
        #                     (shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100 > 2:
        #                 print(f"성능저하 {((performance - prevPerformance) / prevPerformance) * 100}")
        #                 gpuTool.setGpuClock(int(gpuTool.getGpuClock() * 1.1))  # 10퍼센트 올림
        #                 isUp = True
        #
        #
        #             else:
        #                 # 위 조건을 만족하지 않으면 클럭을 더 내리기
        #                 gpuTool.setGpuClock(int(gpuTool.getGpuClock() * 0.9))
        #
        #             # 클럭을 내렸을때 차이가 별로 없다면 이제 그 클럭으로 고정해도 되지않을까?
        #         elif isUp:  # 클럭을 올렸을때
        #             print(f"파워소모차이 {((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw)}")
        #             if ((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100 > 10:
        #                 print(f"파워소모량 증가 {((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100}")
        #                 gpuTool.setGpuClock(int(gpuTool.getGpuClock() * 0.9))
        #                 isUp = False
        #
        #             else:
        #                 # 위 조건을 만족하지 않으면 클럭을 더 올리기
        #                 gpuTool.setGpuClock(int(gpuTool.getGpuClock() * 1.1))

        sumPowerDraw += shared_powerDraw_1epoch.value

        prevPerformance = performance
        prevPowerDraw = shared_powerDraw_1epoch.value
        print(f"걸린 시간{performance}sec")
        print(f"소모 전력 (1에폭) {shared_powerDraw_1epoch.value}")

        shared_powerDraw_1epoch.value = 0  # 1에폭 소모량 초기화

    print("모델 학습 완료.")
    print(f"총 소모 전력 : {sumPowerDraw}")
    print(f"총 걸린 시간 : {sumPerformance}")

if __name__ == "__main__":


    manager = multiprocessing.Manager()
    shared_isStart = manager.Value('b', False)
    shared_powerDraw_1epoch = manager.Value('d', 0.0)

    #profiler_process = multiprocessing.Process(target = updateGpuInfo)
    profiler_process2 = multiprocessing.Process(target=getPowerDrawPerSecond,
                                                args=(shared_isStart, shared_powerDraw_1epoch))
    profiler_process3 = multiprocessing.Process(target=resNet18,
                                                args=(shared_isStart, shared_powerDraw_1epoch))
    #profiler_process.start()
    profiler_process2.start()
    profiler_process3.start()

    # train_cifar10_with_gpu 프로세스가 종료될 때까지 대기
    profiler_process3.join()

    # 나머지 프로세스를 종료
    #profiler_process.terminate()
    profiler_process2.terminate()

    # 프로세스가 종료될 때까지 기다림
    #profiler_process.join()
    profiler_process2.join()

    gpuTool.resetGpuClock()