import React, { useState } from "react";
import logo from "./image.png";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  const [image, setImage] = useState(null);
  const [textQuery, setTextQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [queryType, setQueryType] = useState("text");
  const [retrievalModel, setRetrievalModel] = useState("tfidf");

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
  };

  const handleTextChange = (e) => {
    setTextQuery(e.target.value);
  };

  const handleQueryTypeChange = (type) => {
    setQueryType(type);
    setTextQuery("");
    setImage(null);
  };

  const handleRetrievalModelChange = (e) => {
    setRetrievalModel(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const formData = new FormData();
    formData.append("query_type", queryType);
    formData.append("model", retrievalModel);

    if (textQuery.trim()) {
      formData.append("query_given", textQuery);
    }

    if (image) {
      formData.append("image_file", image);
    } else {
      formData.append("image_file", "None");
    }

    try {
      const response = await fetch("http://localhost:5000/query", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResults(data.results || []);
    } catch (error) {
      console.error("Error fetching results:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleNewQuery = () => {
    setTextQuery("");
    setImage(null);
    setResults([]);
    setQueryType("text");
    setLoading(false);
  };

  return (
    <div className="container my-5">
      <div className="text-center mb-4">
        <img src={logo} alt="Company Logo" className="img-fluid" style={{ maxWidth: "200px" }} />
        <h1>#IRE Project</h1>
      </div>

      <div className="mb-4">
        <ul className="nav nav-tabs">
          <li className="nav-item">
            <button
              className={`nav-link ${queryType === "text" ? "active" : ""}`}
              onClick={() => handleQueryTypeChange("text")}
            >
              Text Query
            </button>
          </li>
          <li className="nav-item">
            <button
              className={`nav-link ${queryType === "text+image" ? "active" : ""}`}
              onClick={() => handleQueryTypeChange("text+image")}
            >
              Text + Image Query
            </button>
          </li>
        </ul>
      </div>

      <form onSubmit={handleSubmit} className="mb-4">
        <div className="mb-3">
          <label className="form-label">Retrieval Model:</label>
          <select
            className="form-select"
            value={retrievalModel}
            onChange={handleRetrievalModelChange}
          >
            <option value="bm25">BM25</option>
            <option value="tfidf">TF-IDF</option>
            <option value="sentencebert">SentenceBERT</option>
          </select>
        </div>

        {queryType.includes("text") && (
          <div className="mb-3">
            <label className="form-label">Enter Text Query:</label>
            <input
              type="text"
              className="form-control"
              placeholder="Enter text query"
              value={textQuery}
              onChange={handleTextChange}
            />
          </div>
        )}

        {queryType.includes("image") && (
          <div className="mb-3">
            <label className="form-label">Upload Image:</label>
            <input
              type="file"
              className="form-control"
              accept="image/*"
              onChange={handleImageChange}
            />
            {image && (
              <div className="mt-3">
                <p>Uploaded Image:</p>
                <img
                  src={URL.createObjectURL(image)}
                  alt="Uploaded"
                  className="img-thumbnail"
                  style={{ maxWidth: "200px" }}
                />
              </div>
            )}
          </div>
        )}

        <button type="submit" className="btn btn-primary w-100" disabled={loading}>
          {loading ? "Processing..." : "Submit Query"}
        </button>
      </form>

      <button onClick={handleNewQuery} className="btn btn-secondary w-100 mb-4">
        New Query
      </button>

      {results.length > 0 && (
        <div>
          <h2 className="mb-4">Top Results:</h2>
          <div className="row">
            {results.map((item, index) => (
              <div className="col-md-4 mb-3" key={index}>
                <div className="card h-100">
                  <img
                    src={item.image_url || "https://via.placeholder.com/200"}
                    className="card-img-top"
                    alt={`Result ${index + 1}`}
                  />
                  <div className="card-body">
                    <h5 className="card-title">{item.title}</h5>
                    <p className="card-text">
                      {item.item_specifics
                        ? Object.entries(item.item_specifics).map(([key, value]) => (
                            <div key={key}>
                              <strong>{key}:</strong> {value}
                            </div>
                          ))
                        : "No specifics available."}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
