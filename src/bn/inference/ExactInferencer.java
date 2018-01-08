package bn.inference;

import java.util.List;
import java.io.*;
import javax.xml.parsers.ParserConfigurationException;
import org.xml.sax.SAXException;

import bn.core.*;
import bn.parser.*;

public class ExactInferencer implements Inferencer {
	
	public static void main(String[] args){
		ExactInferencer infer = new ExactInferencer();
		
		if(args[0].contains(".xml")){
			XMLBIFParser x = new XMLBIFParser();
			try {
				BayesianNetwork bn = x.readNetworkFromFile(args[0]);
				Assignment e = new Assignment();
				for(int i = 2; i < args.length; i+=2){
					e.put(bn.getVariableByName(args[i]), args[i+1]);
				}
				Distribution dist = infer.ask(bn, bn.getVariableByName(args[1]), e);
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
				x = new BIFParser(new FileInputStream(args[0]));
				//System.out.println(x.parseNetwork());
				BayesianNetwork bn = x.parseNetwork();
				Assignment e = new Assignment();
				for(int i = 2; i < args.length; i+=2){
					e.put(bn.getVariableByName(args[i]), args[i+1]);
				} 
				Distribution dist = infer.ask(bn, bn.getVariableByName(args[1]), e);
				
				System.out.println(dist);
			} catch (FileNotFoundException e1) {
				e1.printStackTrace();
			} catch (IOException e1) {
				e1.printStackTrace();
			}
		}
		
	}
	
	/**
	 * The enumeration algorithm for answering queries on Bayesian networks.  
	 * From AIMA Figure 14.9
	 */
	
	//ENUMERATOR-ASK
	public Distribution ask(BayesianNetwork bn, RandomVariable X, Assignment e) {
		
		//initially empty distribution of X
		Distribution Q = new Distribution(X);
		
		for(int i = 0; i < X.getDomain().size(); i++) {  
			//pass the assignment in the clone to avoid pointer error
			Assignment copy = e.copy();
			copy.set(X, X.getDomain().get(i));
			Q.put(X.getDomain().get(i), enumerateAll(bn, copy, 0));
		}
		Q.normalize();
		return Q;
	}
	
	//ENUMERATOR-ALL
	//add an index parameter to enumerate through the vars list without computing REST(vars)
	public double enumerateAll(BayesianNetwork bn, Assignment e, int index) {
		
		//the list of variables from the Bayesian network must be topological sorted
		//the pseudo-code from the figure do not mention this
		List<RandomVariable> vars = bn.getVariableListTopologicallySorted();
		
		//return 1 if vars is empty
		if(index >= vars.size()) {
			return 1;
		}
		
		RandomVariable Y = vars.get(index);	
		
		//if y has a value in the Bayesian network
		if(e.containsKey(Y)) {
			e = e.copy();
			return bn.getProb(Y, e)*enumerateAll(bn, e, index+1);
		} else {
			double sum = 0;
			
			//for loop to compute the sum of products of conditional probabilities i the network
			for(int i = 0; i < Y.getDomain().size(); i++){
				e.put(Y, Y.getDomain().get(i));
				Assignment ni = e.copy();
				double prob = bn.getProb(Y, ni);
				double enumAll = enumerateAll(bn, ni, index+1);
				sum += prob*enumAll;
			}
			
			return sum;
		}
	}
}