// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBYc1K57qqyIBHp1K7ZlKnZvJDpZzEzK4A",
  authDomain: "gpudashboa.firebaseapp.com",
  projectId: "gpudashboa",
  storageBucket: "gpudashboa.appspot.com",
  messagingSenderId: "128148028990",
  appId: "1:128148028990:web:805bdd8cc255bf46e6db6f",
  measurementId: "G-ZXMQG8Z2P3"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);