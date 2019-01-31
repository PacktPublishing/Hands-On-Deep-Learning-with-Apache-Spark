package org.googlielmo.vgg16webservice;

import static spark.Spark.port;
import static spark.Spark.post;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import javax.servlet.MultipartConfigElement;
import javax.servlet.ServletException;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import com.google.gson.Gson;

import spark.Request;

public class WebServer {
	private void loadModel() {
		// Change the listening port
	    port(8998);
	    
	    // upload endpoint
	    post("/upload", (req, res) -> uploadFile(req));
	    //Gson gson = new Gson();
	    //post("/upload", (req, res) -> uploadFile(req), gson::toJson);
	}
	
	private String uploadFile(Request req) throws IOException, ServletException {
		// Create the upload directory
		File uploadDir = new File("upload");
		uploadDir.mkdir(); 
		
		// Load the serialized model
		ClassLoader classLoader = getClass().getClassLoader();
		File serializedModelFile = new File(classLoader.getResource("Vgg-16.zip").getFile());
		ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(serializedModelFile);
				
		Path tempFile = Files.createTempFile(uploadDir.toPath(), "", "");

        req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));

        try (InputStream input = req.raw().getPart("file").getInputStream()) { 
          Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING);
        }

        // The user submitted file is tempFile, convert to Java File "file"
        File file = tempFile.toFile();

        // Convert file to INDArray
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(file);

        // delete the physical file, if left our drive would fill up over time
        file.delete();

        // Mean subtraction pre-processing step for VGG
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);

        //Inference returns array of INDArray, index[0] has the predictions
        INDArray[] output = vgg16.output(false,image);

        ImageNetLabels imagNetLabels = new ImageNetLabels();
        String predictions = imagNetLabels.decodePredictions(output[0]);
        
		return predictions;
	}
	
	public static void main(String[] args) throws Exception {
		WebServer server = new WebServer();
		server.loadModel();
    }
}
