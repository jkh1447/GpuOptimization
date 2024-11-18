import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import time
import gpuTool
from sendGpuInfo import updateGpuInfo
import multiprocessing
from tqdm import tqdm

def getPowerDrawPerSecond(shared_isStart, shared_powerDraw_1epoch):
    sumPowerDraw = 0

    while True:
        if (shared_isStart):
            sumPowerDraw = gpuTool.getGpuPowerDraw()
            shared_powerDraw_1epoch.value += float(sumPowerDraw)
            time.sleep(1)

# VGGNet 정의
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def train_model(shared_isStart, shared_powerDraw_1epoch):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = VGGNet().to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # max clock
    max_Clock = 1950
    prevPerformance = None
    prevPowerDraw = None
    isUp = False
    max_Clock = 1950

    sumPowerDraw = 0
    sumPerformance = 0

    # 모델 학습
    for epoch in range(5):  # 데이터셋을 여러번 반복

        start = time.time()
        shared_isStart = True  # 파워소모측정 시작

        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{5}', total=len(trainloader), leave=False)

        for i, data in enumerate(progress_bar):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:  # 매 2000 미니배치마다 출력
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            progress_bar.set_postfix(loss=running_loss / (i + 1))  # 평균 손실 업데이트

        end = time.time()
        shared_isStart = False  # 파워소모 측정 종료
        performance = end - start

        sumPerformance += performance

        # default가 최고성능, 첫번째 에폭때엔 클럭 -30%
        # if epoch == 0:
        #     print(max_Clock * 0.8)
        #     best_performance = performance
        #     gpuTool.setGpuClock(int(max_Clock * 0.8))
        #     isUp = False
        # else:
        #     if ((abs(((performance - prevPerformance) / prevPerformance) * 100) < 2) and (
        #             abs(((performance - prevPerformance) / prevPerformance) * 100) > 0)) or \
        #             ((abs(((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100) < 2) and \
        #              (abs(((shared_powerDraw_1epoch.value - prevPowerDraw) / prevPowerDraw) * 100) > 0)):
        #         # 성능저하차이도 적고, 파워소모 차이도 적으면 그 클럭을 계속 유지
        #         print("클럭을 결정")
        #         isClockDone = True
        #     if not isClockDone:
        #         if not isUp:  # 클럭을 내렸을때
        #             print(f"성능차이{((performance - prevPerformance) / prevPerformance) * 100}")
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

    #profiler_process = multiprocessing.Process(target=updateGpuInfo)
    profiler_process2 = multiprocessing.Process(target=getPowerDrawPerSecond,
                                                args=(shared_isStart, shared_powerDraw_1epoch))
    profiler_process3 = multiprocessing.Process(target=train_model,
                                                args=(shared_isStart, shared_powerDraw_1epoch))
    #profiler_process.start()
    profiler_process2.start()
    profiler_process3.start()


    profiler_process3.join()

    # 나머지 프로세스를 종료
    #profiler_process.terminate()
    profiler_process2.terminate()

    # 프로세스가 종료될 때까지 기다림
    #profiler_process.join()
    profiler_process2.join()

    gpuTool.resetGpuClock()