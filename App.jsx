import './App.css'
import { useState } from 'react'
import './orm.css'
function App() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [name, setName] = useState("");
    const [roll, setRoll] = useState("");

    const handleFileChange = (e) => setFile(e.target.files[0]);

    const handleUpload = async () => {
      if (!file) return alert("Please select a file!");

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) throw new Error("Upload failed");

        const data = await res.json();
        setResult(data);
      } catch (err) {
        console.error(err);
        alert("Upload error: " + err.message);
      }
    };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-2xl font-bold mb-4">Automated OMR Evaluation</h1>
     <label>Enter Full Name</label>
      <input
        type="text"
        placeholder="Enter Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="mb-2 p-2 border rounded w-full max-w-md"
      />
      <label >Enter your Roll Number</label>
      <input
        type="text"
        placeholder="Enter Roll Number"
        value={roll}
        onChange={(e) => setRoll(e.target.value)}
        className="mb-4 p-2 border rounded w-full max-w-md"
      />
      <label >Upload Your ORM-Sheet</label>
      <input type="file" onChange={handleFileChange} className="mb-4" />

      <button
        onClick={handleUpload}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 mb-4"
      >
        Upload & Evaluate
      </button>

      <div className="box">
        {result && (
          <div className="bg-white shadow p-4 rounded w-full max-w-md">
            <h2 className="font-semibold mb-2">Result:</h2>
            <div className="data">
              <table>
                <tr>
                  <td>
                    <strong>Name:</strong>
                  </td>
                  <td>{name}</td>
                </tr>

                <tr>
                  <td>
                    <strong>Roll Number:</strong>
                  </td>
                  <td>{roll}</td>
                </tr>
                <tr>
                  <td>
                    <strong>Score:</strong>
                  </td>
                  <td>{result.score}</td>
                </tr>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App
