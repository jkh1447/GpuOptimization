import subprocess



def getGpuClock():
    t_clock = subprocess.run(['nvidia-smi', '--query-gpu=clocks.current.graphics',
                              '--format=csv,noheader,nounits'],
                             stdout=subprocess.PIPE)
    clock = t_clock.stdout.decode('utf-8').strip()
    #print(f"Current Clock : {clock}")

    return int(clock)

def getGpuTemp():
    t_temperature = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
                                    '--format=csv,noheader,nounits'],
                                   stdout=subprocess.PIPE)
    temperature = t_temperature.stdout.decode('utf-8').strip()
    #print(f"Current Temperature : {temperature}")

    return temperature

def getGpuPowerDraw():
    t_powerDraw = subprocess.run(['nvidia-smi', '--query-gpu=power.draw',
                                  '--format=csv,noheader,nounits'],
                                 stdout=subprocess.PIPE)
    powerDraw = t_powerDraw.stdout.decode('utf-8').strip()
    #print(f"powerDraw : {powerDraw}")

    return powerDraw

# gpu 클럭 온도 파워소모량(W)
def getGpuInfo():
    try:

        t_clock = subprocess.run(['nvidia-smi', '--query-gpu=clocks.current.graphics',
                                 '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE)
        t_temperature = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
                                 '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE)
        t_powerDraw = subprocess.run(['nvidia-smi', '--query-gpu=power.draw',
                                 '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE)

        clock = t_clock.stdout.decode('utf-8').strip()
        temperature = t_temperature.stdout.decode('utf-8').strip()
        powerDraw = t_powerDraw.stdout.decode('utf-8').strip()

        print(f"Current Clock : {clock}")
        print(f"Current Temperature : {temperature}")
        print(f"powerDraw : {powerDraw}")



        return [clock, temperature, powerDraw]

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")

# Gpu클럭 설정
def setGpuClock(clock):
    try:
        # nvidia-smi 명령어 실행
        result = subprocess.run(
            ['nvidia-smi', '-lgc', f'{clock}'], capture_output=True, text=True, check=True)

        # 출력 결과 표시
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")

# 기본 Gpu클럭 설정
def resetGpuClock():
    try:
        # nvidia-smi 명령어 실행
        result = subprocess.run(
            ['nvidia-smi', '-rgc'], capture_output=True, text=True, check=True)

        # 출력 결과 표시
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")

