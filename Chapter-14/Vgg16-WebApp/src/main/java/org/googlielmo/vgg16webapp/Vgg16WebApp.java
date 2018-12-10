package org.googlielmo.vgg16webapp;

import static spark.Spark.get;
import static spark.Spark.post;
import static spark.Spark.port;
import static spark.Spark.staticFiles;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import javax.servlet.MultipartConfigElement;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

public class Vgg16WebApp {
	public void loadModel() throws IOException {
		// Load the serialized model
		ClassLoader classLoader = getClass().getClassLoader();
		File serializedModelFile = new File(classLoader.getResource("Vgg-16.zip").getFile());
		ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(serializedModelFile);
		
		// Create the upload directory
		File uploadDir = new File("upload");
	    uploadDir.mkdir(); 
	    
	    // Implement the page header
	    String header = buildFoundationHeader();
	    
	    // Implement the upload form
	    String form = buildUploadForm();
	    
	    staticFiles.location("/public");
	    
	    // Change the listening port
	    port(8998);
	    
	    // Set the route for the prediction page
	    get("Vgg16Predict", (req, res) -> header + form);
	    
	    post("/getPredictions", (req, res) -> {

	        Path tempFile = Files.createTempFile(uploadDir.toPath(), "", "");

	        req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));

	        try (InputStream input = req.raw().getPart("uploadedFile").getInputStream()) { 
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

	        // convert 1000 length numeric index of probabilities per label
	        // to sorted return top 5 convert to string using helper function VGG16.decodePredictions
	        // "predictions" is string of our results
	        //String predictions = TrainedModels.VGG16.decodePredictions(output[0]);
	        // String predictions = output[0].data().toString();
	        
	        ImageNetLabels imagNetLabels = new ImageNetLabels();
	        String predictions = imagNetLabels.decodePredictions(output[0]);
	        
	        // return results along with form to run another inference
	        return buildFoundationHeader() + "<h4> '" + predictions + "' </h4>" +
	          "Would you like to try with another image?" +
	          form;
	        //return "<h1>Your image is: '" + tempFile.getName(1).toString() + "' </h1>";

	      });
	    
	}
	
	private String buildFoundationHeader() {
		String header = "<head>\n"
				+ "<link rel='stylesheet' href='foundation-float.min.css'>\n"
				+ "</head>\n";
		
		return header;
	}
	
	private String buildUploadForm() {
		String form = 
				  "<form method='post' action='getPredictions' enctype='multipart/form-data'>\n" +
			      " <input type='file' name='uploadedFile'>\n" +
			      " <button class='success button'>Upload picture</button>\n" +
			      "</form>\n";
		
		return form;
	}
	
	public static void main(String[] args) throws Exception {
		Vgg16WebApp webApp = new Vgg16WebApp();
		webApp.loadModel();
		
		//get("/hello", (req, res) -> "Hello VGG16");
	}
}
