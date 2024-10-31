import React, {useState, useEffect} from 'react';
import {View, ScrollView, Button, Text, TextInput, Image, StyleSheet, Dimensions} from 'react-native';
import {SpeedoMeterPlus} from 'react-native-speedometer-plus';
const {initializeApp} = require("firebase/app");
const {getDatabase, ref, set, onValue} = require("firebase/database");
import { LineChart, BarChart  } from 'react-native-chart-kit';
import Speedometer, {Background, Arc, Needle, Progress, Marks, Indicator} from 'react-native-cool-speedometer';


export default function MyLayout(){

    const firebaseConfig = {
        apiKey: "AIzaSyBYc1K57qqyIBHp1K7ZlKnZvJDpZzEzK4A",
        authDomain: "gpudashboa.firebaseapp.com",
        databaseURL: "https://gpudashboa-default-rtdb.asia-southeast1.firebasedatabase.app",
        projectId: "gpudashboa",
        storageBucket: "gpudashboa.appspot.com",
        messagingSenderId: "128148028990",
        appId: "1:128148028990:web:805bdd8cc255bf46e6db6f",
        measurementId: "G-ZXMQG8Z2P3"
      };

    const firebase = initializeApp(firebaseConfig);
    const database = getDatabase(firebase);

    const ClockLabels = [
        {
            name: 'Test1',
            labelColor: '#3b699e',
            activeBarColor: '#3b699e',
          },
          
          
          
          
    ]
    const TempLabels = [
        {
            name: 'Test1',
            labelColor: '#ff5153',
            activeBarColor: '#ff5153',
          },
          
    ]
    const PowerLabels = [
        {
            name: 'Test1',
            labelColor: '#02a361',
            activeBarColor: '#02a361',
          },
          
    ]

    const [inputValue, setInputValue] = useState('');

    const [clock, setClock] = useState(0);
    const [temp, setTemp] = useState(0);
    const [power, setPower] = useState(0);
    const [clockChartData, setClockChartData] = useState([0]);
    const [tempChartData, setTempChartData] = useState([0]);
    const [powerChartData, setPowerChartData] = useState([0]);

    const [clockChartLen, setClockChartLen] = useState(0);
    const [tempChartLen, setTempChartLen] = useState(0);
    const [powerChartLen, setPowerChartLen] = useState(0);

    const [defaultPower, setDefaultPower] = useState(0);
    const [OptiPower, setOptiPower] = useState(0);
    
    

    useEffect(()=> {
        const firstRef = ref(database, 'firstEpoch')
        const firstListener = onValue(firstRef, (snapshot) => {
            console.log("firstListener")
            const data = snapshot.val();
            const done = data.done
            if (done) {
                setDefaultPower(parseInt(data.powerDraw * data.epoch))
            }


                
        })

        const epochRef = ref(database, 'epochInfo')
        const epochListener = onValue(epochRef, (snapshot) => {
            console.log("epoch")
            const data = snapshot.val();
            setOptiPower(prev => prev + data.powerDraw)
        })

        const gpuLogsRef = ref(database, 'gpuLogs');

        const logListener = onValue(gpuLogsRef, (snapshot) => {
            const data = snapshot.val();
            const newClock = parseInt(data.clock, 10);
            const newTemp = parseInt(data.temperature, 10);
            const newPower = parseInt(data.powerConsumption, 10);
            setClock(newClock);
            setTemp(newTemp);
            setPower(newPower);

            setClockChartData(prevData => {
                const updatedData = [...prevData, newClock];
                setClockChartLen(prevLen => prevLen + 1); 
                
                return updatedData.length > 20 ? updatedData.slice(1) : updatedData;
                //return updatedData;
            });

            setTempChartData(prevData => {
                const updatedData = [...prevData, newTemp];
                setTempChartLen(prevLen => prevLen + 1);
                return updatedData.length > 20 ? updatedData.slice(1) : updatedData;
                //return updatedData;
            });

            setPowerChartData(prevData => {
                const updatedData = [...prevData, newPower];
                setPowerChartLen(prevLen => prevLen + 1);
                return updatedData.length > 20 ? updatedData.slice(1) : updatedData;
                //return updatedData;
            });


            //console.log(`clock : ${newClock}`);
        });

        return () => {
            logListener();
            firstListener();
            epochListener();
        };
    }, [])
   
  return (
    

        <View style={styles.container}>
            {/* 제목 구역 */}
            <View style={styles.titleBox}>
                <Text style={styles.titleText}>GPU Dashboard</Text>
            </View>
        {/* 상단 4개의 구역 */}
        <View style={styles.topContainer}>
            <View style={styles.box}>
                <SpeedoMeterPlus
                                value={clock}
                                size={200}  
                                minValue={0}
                                maxValue={3000}
                                innerLabelNoteValue="CLOCK"
                                labels = {ClockLabels}
                                innerLabelNoteStyle={{ color: '#3b699e', fontSize: 23 }}
                                
                                
                    />
                

            </View>
            <View style={styles.box}>
            <SpeedoMeterPlus
                            value={temp}
                            size={200}
                            minValue={0}
                            maxValue={100}
                            innerLabelNoteValue="TEMP"
                            labels = {TempLabels}
                            innerLabelNoteStyle={{ color: '#ff5153', fontSize: 23 }}
                        />
            </View>
            <View style={styles.box}>
            <SpeedoMeterPlus
                            value={power}
                            size={200}
                            minValue={0}
                            maxValue={200}
                            innerLabelNoteValue="POWER"
                            labels = {PowerLabels}
                            innerLabelNoteStyle={{ color: '#02a361', fontSize: 23 }}
                        />
            </View>
            <View style={styles.infoBox}>
                <BarChart
                    data={{
                        labels: ['Default', 'Optimized'], // X축 레이블
                        datasets: [
                          {
                            data: [defaultPower, OptiPower], // Y축 값
                          },
                        ],
                      }}
                    width={200} // 화면 너비에서 여백을 뺀 값
                    height={220} // 차트 높이
                    yAxisLabel="" // Y축 레이블
                    chartConfig={{
                    backgroundColor: '#ffffff', // 배경색
                    backgroundGradientFrom: '#ffffff', // 그라데이션 시작색
                    backgroundGradientTo: '#ffffff', // 그라데이션 끝색
                    decimalPlaces: 2, // 소수점 자리수
                    color: (opacity = 0) => `rgba(102, 255, 51, ${opacity})`, // 바 색상
                    labelColor: (opacity = 1) => `rgba(59, 105, 158, ${opacity})`, // 레이블 색상
                    style: {
                        borderRadius: 16, // 테두리 반경
                    },
                    propsForDots: {
                        r: '6',
                        strokeWidth: '2',
                        stroke: '#ffa726',
                    },
                    }}
                    style={{
                    marginVertical: 8,
                    borderRadius: 16,
                    
                    }}
                    
                    fromZero={true} // 시작점을 0으로 설정
                />
                {/* 수직으로 텍스트 배치 */}
                <View style={styles.textContainer}>
                    <Text style={styles.infoText}>Default {defaultPower}W</Text>
                    <Text>{'\n'}</Text>
                    <Text>{'\n'}</Text>
                    <Text style={styles.infoText}>Optimized {parseInt(OptiPower)}W</Text>
                </View>
            </View>
        </View>
        
        {/* 하단 큰 구역 */}
        <View style={styles.bottomContainer}>
            <View style={styles.box}>
                <View style={styles.chartContainer}>
                    
                    <Text style={styles.chartTitle}>Clock</Text>
                        <LineChart
                            data={{
                                labels: clockChartLen > 20
                                ? Array.from({ length: 20 }, (_, index) => (clockChartLen - 20 + index + 1).toString())
                                : clockChartData.map((_, index) => (index + 1).toString()), // 기본 레이블 1부터 시작
                                datasets: [
                                    {
                                        data: clockChartData,
                                        strokeWidth: 4,
                                    },
                                ],
                            }}
                            width={400} // react-native에서
                            height={400}
                            yAxisLabel=""
                            yAxisSuffix=" Hz"
                            chartConfig={{
                                backgroundColor: '#ffffff',
                                backgroundGradientFrom: '#ffffff',
                                backgroundGradientTo: '#ffffff',
                                decimalPlaces: 0,
                                color: (opacity = 1) => `rgba(59, 105, 158, ${opacity})`,
                                labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                                style: {
                                    borderRadius: 16,
                                    borderWidth: 0,
                                },
                                propsForDots: {
                                    r: 0, // 동그라미 표시 제거
                                    fill: 'none', // 색상 없애기
                                },
                                useShadowColorFromDataset: true,
                                fillShadowGradientFrom: '#ffffff', // 투명 음영
                                fillShadowGradientOpacity: 0, // 음영 투명도 0으로 설정
                            }}
                            style={styles.chartStyle}
                        />
                    
                </View>
            </View>
            <View style={styles.box}>
                <View style={styles.chartContainer}>
                <Text style={styles.chartTitle}>Temp</Text>
                <LineChart
                        data={{
                            labels: tempChartLen > 20
                            ? Array.from({ length: 20 }, (_, index) => (tempChartLen - 20 + index + 1).toString())
                            : tempChartData.map((_, index) => (index + 1).toString()), // 기본 레이블 1부터 시작
                            datasets: [
                                {
                                    data: tempChartData,
                                    strokeWidth: 4,
                                },
                            ],
                        }}
                        width={400} // react-native에서
                        height={400}
                        yAxisLabel=""
                        yAxisSuffix="°C"
                        yAxisInterval={10} // y축 간격 설정 (예: 1)
                        chartConfig={{
                            backgroundColor: '#ffffff',
                            backgroundGradientFrom: '#ffffff',
                            backgroundGradientTo: '#ffffff',
                            decimalPlaces: 0,
                            color: (opacity = 1) => `rgba(255, 81, 83, ${opacity})`,
                            labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                            style: {
                                borderRadius: 16,
                            },
                            propsForDots: {
                                r: 0, // 동그라미 표시 제거
                                fill: 'none', // 색상 없애기
                            },
                            useShadowColorFromDataset: true,
                            fillShadowGradientFrom: '#ffffff', // 투명 음영
                            fillShadowGradientOpacity: 0, // 음영 투명도 0으로 설정
                        }}
                        style={styles.chartStyle}
                    />
                </View>    
            </View>
            
            <View style={styles.box}>
                <View style={styles.chartContainer}>
                <Text style={styles.chartTitle}>Power</Text>
                    <LineChart
                            data={{
                                labels: powerChartLen > 20
                                ? Array.from({ length: 20 }, (_, index) => (powerChartLen - 20 + index + 1).toString())
                                : powerChartData.map((_, index) => (index + 1).toString()), // 기본 레이블 1부터 시작
                                datasets: [
                                    {
                                        data: powerChartData,
                                        strokeWidth: 4,
                                    },
                                ],
                            }}
                            width={400} // react-native에서
                            height={400}
                            yAxisLabel=""
                            yAxisSuffix="W"
                            chartConfig={{
                                backgroundColor: '#ffffff',
                                backgroundGradientFrom: '#ffffff',
                                backgroundGradientTo: '#ffffff',
                                decimalPlaces: 0,
                                color: (opacity = 1) => `rgba(2, 163, 97, ${opacity})`,
                                labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                                style: {
                                    borderRadius: 16,
                                },
                                propsForDots: {
                                    r: 0, // 동그라미 표시 제거
                                    fill: 'none', // 색상 없애기
                                },
                                useShadowColorFromDataset: true,
                                fillShadowGradientFrom: '#ffffff', // 투명 음영
                                fillShadowGradientOpacity: 0, // 음영 투명도 0으로 설정
                            }}
                            style={styles.chartStyle}
                        />
                </View>
            </View>
                    
            <View style={styles.logoBox}></View>


        </View>
        </View>
    
  );
};

const styles = StyleSheet.create({
    container: { flex: 1, padding: 20, backgroundColor: '#e0e0e0' },
    titleBox: { paddingVertical: 15, backgroundColor: '#fff', marginBottom: 15, borderRadius: 10, alignItems: 'flex-start', borderTopWidth: 2, // 위쪽 테두리 두께 설정
        borderTopColor: '#53a3ec',    marginHorizontal: 14,},
    titleText: { fontSize: 24, fontWeight: 'bold', color: '#3b699e', marginLeft : 10 },
    topContainer: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 20 },
    bottomContainer : {flexDirection: 'row', justifyContent: 'space-between', marginBottom: 20},
    box: { flex: 1, backgroundColor: '#fff', padding: 15, borderRadius: 10, alignItems: 'center', marginHorizontal: 7, paddingTop: 50,borderTopWidth: 2, // 위쪽 테두리 두께 설정
        borderTopColor: '#53a3ec',},
    logoBox:{flex : 0.8, backgroundColor: '#fff', padding: 15, borderRadius: 10, alignItems: 'center', marginHorizontal: 7, paddingTop: 50,borderTopWidth: 2, // 위쪽 테두리 두께 설정
        borderTopColor: '#53a3ec',},
    infoBox: { flex: 1.5, backgroundColor: '#fff', borderRadius: 10, padding: 15, alignItems: 'center', marginHorizontal: 5, flexDirection:'row', borderTopWidth: 2, // 위쪽 테두리 두께 설정
        borderTopColor: '#53a3ec', },
    infoText: { fontSize: 16, color: '#333' },
    bottomBox: { flex: 2, backgroundColor: '#fff', padding: 15, borderRadius: 10 },
    chartContainer: { marginRight: 20, alignItems: 'center' },
    chartTitle: { fontSize: 18, fontWeight: '600', color: '#333', marginBottom: 5 },
    chartStyle: { marginVertical: 8, borderRadius: 16 },
    labelStyle: { fontSize: 18, color: '#333' },
    textContainer: {flexDirection:'column', alignItems:'flex-start', marginLeft:70},
    infoText: { 
        fontSize: 30, 
        fontWeight: 'bold',
        color: '#02a361',
        marginVertical: 4, // 각 텍스트 간의 여백
      },
});
