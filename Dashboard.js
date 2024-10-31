import {View, ScrollView, Button, Text, TextInput, Image, SafeAreaView} from 'react-native';
import React, {useState, useEffect} from 'react';
import {SpeedoMeterPlus} from 'react-native-speedometer-plus';
const {initializeApp} = require("firebase/app");
const {getDatabase, ref, set, onValue} = require("firebase/database");

export default function Dashboard(){
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


    const allLabels = [
        {
            name: 'Test1',
            labelColor: 'gray',
            activeBarColor: 'gray',
          },
          {
            name: 'Test2',
            labelColor: 'gray',
            activeBarColor: 'gray',
          },
          {
            name: 'Test3',
            labelColor: 'gray',
            activeBarColor: 'gray',
          },
    ]

    const [inputValue, setInputValue] = useState('');

    const [clock, setClock] = useState(0);
    const [temp, setTemp] = useState(0);
    const [power, setPower] = useState(0);

    useEffect(()=> {
        const gpuLogsRef = ref(database, 'gpuLogs');

        const logListener = onValue(gpuLogsRef, (snapshot) => {
            const data = snapshot.val();
            setClock(parseInt(data.clock, 10));
            setTemp(parseInt(data.temperature, 10));
            setPower(parseInt(data.powerConsumption, 10));
            console.log("changed");
        });

        return () => logListener();
    }, [])

    // const handlePress = () => {
    //     const parsedNumber = parseInt(inputValue, 10);
    //     if (!isNaN(parsedNumber)) {
    //         setNumber(parsedNumber);
    //     } else {
    //         alert('유효한 숫자를 입력하세요.');
    //     }
    // }


    return (
        
            <ScrollView showsVerticalScrollIndicator={false}>

                <SafeAreaView style={{ flexDirection:'row', justifyContent : 'space-between',}}>
                    {/* <TextInput
                        value={inputValue}
                        onChangeText={setInputValue}
                    />
                    <Button title="적용" onPress={handlePress}/> */}

                    <SpeedoMeterPlus
                        value={clock}
                        size={200}  
                        minValue={0}
                        maxValue={3000}
                        innerLabelNoteValue="CLOCK"
                        labels = {allLabels}
                        innerLabelNoteStyle={{ color: 'black', fontSize: 23 }}
                        
                        
                    />
                    <SpeedoMeterPlus
                        value={temp}
                        size={200}
                        minValue={0}
                        maxValue={100}
                        innerLabelNoteValue="TEMP"
                        labels = {allLabels}
                        innerLabelNoteStyle={{ color: 'black', fontSize: 23 }}
                    />
                    <SpeedoMeterPlus
                        value={power}
                        size={200}
                        minValue={0}
                        maxValue={100}
                        innerLabelNoteValue="POWER"
                        labels = {allLabels}
                        innerLabelNoteStyle={{ color: 'black', fontSize: 23 }}
                    />
                    
                </SafeAreaView>
            </ScrollView>
        
    );
}