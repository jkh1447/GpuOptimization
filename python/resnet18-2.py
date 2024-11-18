import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
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


def train_cifar10_with_gpu(shared_isStart, shared_powerDraw_1epoch):
    # 데이터셋 불러오기 및 전처리
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # ResNet-18 모델 불러오기
    model = models.resnet18(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10의 클래스 수는 10개

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    num_epochs = 5

    # max clock
    max_Clock = 1950

    best_performance = None
    performance = None
    prevPerformance = None
    prevPowerDraw = None
    isUp = False
    isClockDone = False

    sumPowerDraw = 0
    sumPerformance = 0

    for epoch in range(num_epochs):
        # 학습 단계
        start=time.time() # 성능 측정 시작
        shared_isStart = True # 파워소모측정 시작

        model.train()
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{5}', total=len(trainloader), leave=False)

        #배치 학습
        for i, data in enumerate(progress_bar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            if i % 100 == 99:  # 매 100 미니배치마다 출력
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0
            progress_bar.set_postfix(loss=running_loss / (i + 1))  # 평균 손실 업데이트

        end=time.time()
        shared_isStart = False # 파워소모 측정 종료
        performance = end - start

        # 성능 누적
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

        shared_powerDraw_1epoch.value = 0 # 1에폭 소모량 초기화

    print(f"총 소모 전력 : {sumPowerDraw}")
    print(f"총 걸린 시간 : {sumPerformance}")

if __name__ == "__main__":


    manager = multiprocessing.Manager()
    shared_isStart = manager.Value('b', False)
    shared_powerDraw_1epoch = manager.Value('d', 0.0)

    #profiler_process = multiprocessing.Process(target = updateGpuInfo)
    profiler_process2 = multiprocessing.Process(target=getPowerDrawPerSecond, args=(shared_isStart, shared_powerDraw_1epoch))
    profiler_process3 = multiprocessing.Process(target=train_cifar10_with_gpu, args=(shared_isStart, shared_powerDraw_1epoch))

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
