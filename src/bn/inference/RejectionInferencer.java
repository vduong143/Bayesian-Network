package bn.inference;

import java.util.*;
import java.io.*;
import javax.xml.parsers.ParserConfigurationException;
import org.xml.sax.SAXException;

import bn.core.*;
import bn.parser.*;

public class RejectionInferencer {
	
	public static void main(String[] argv){
		RejectionInferencer wat = new RejectionInferencer();
		if(argv[1].contains(".xml")){
			XMLBIFParser x = new XMLBIFParser();
			try {
				int samples = Integer.parseInt(argv[0]);
				BayesianNetwork bn = x.readNetworkFromFile(argv[1]);
				Assignment e = new Assignment();
				for(int i = 3; i < argv.length; i+=2){
					e.put(bn.getVariableByName(argv[i]), argv[i+1]);
				}
				Distribution dist = wat.ask(samples,bn, bn.getVariableByName(argv[2]), e);
				System.out.println(dist);
			} catch (IOException e) {
				e.printStackTrace();
			} catch (ParserConfigurationException e) {
				e.printStackTrace();
			} catch (SAXException e) {
				e.printStackTrace();
			}
		} else {
			BIFParser x;
			try {
				x = new BIFParser(new FileInputStream(argv[1]));
				int samples = Integer.parseInt(argv[0]);
				//System.out.println(x.parseNetwork());
				BayesianNetwork bn = x.parseNetwork();
				Assignment e = new Assignment();
				for(int i = 3; i < argv.length; i+=2){
					e.put(bn.getVariableByName(argv[i]), argv[i+1]);
				} 
				Distribution dist = wat.ask(samples, bn, bn.getVariableByName(argv[2]), e);
				
				System.out.println(dist);
			} catch (FileNotFoundException e1) {
				e1.printStackTrace();
			} catch (IOException e1) {
				e1.printStackTrace();
			}
		}
	}

	public Distribution ask(int N, BayesianNetwork bn, RandomVariable X, Assignment e) {
		Distribution counts = new Distribution(X);
		counts.initialize(X);
		for(int i = 0; i < N; i++){
			Assignment x = priorSample(bn);
			Set<Map.Entry<RandomVariable, Object>> evidence = e.entrySet();
 
            boolean flag = true;
            
			for(Map.Entry<RandomVariable, Object> var : evidence){
				if(!var.getValue().equals(x.get(var.getKey()))){
					flag = false;
					break;
				} 
				
			}
			if(flag){
				counts.put(x.get(X), counts.get(x.get(X))+1);
			}

		}
		counts.normalize();
		return counts;
	}

	public Assignment priorSample(BayesianNetwork bn){
		Assignment x = new Assignment();
		List<RandomVariable> sortedList = bn.getVariableListTopologicallySorted();
		
		for(int i = 0; i < bn.size(); i++){ //for each random variable in bn, in topological order
        //so we grabbed a random variable X[i], set its value weighted randomly given parents 
			RandomVariable Xi = sortedList.get(i);
			ArrayList<Double> weights = new ArrayList<Double>(); 
			for(int j = 0; j < Xi.getDomain().size(); j++){
				x.set(Xi, Xi.getDomain().get(j));
				weights.add(j, bn.getCPTForVariable(Xi).get(x));
			}
			double r = Math.random();
			double sum = 0;
			for(int k = 0; k < weights.size(); k++){
				sum += weights.get(k);
				if(r <= sum){
					x.put(Xi, Xi.getDomain().get(k));
					break;
				}
			} 
		}
		
		return x;
	}
}
