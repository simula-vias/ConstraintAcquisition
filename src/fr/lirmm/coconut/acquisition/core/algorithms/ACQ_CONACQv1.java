package fr.lirmm.coconut.acquisition.core.algorithms;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import fr.lirmm.coconut.acquisition.core.acqconstraint.ACQ_CNF;
import fr.lirmm.coconut.acquisition.core.acqconstraint.*;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ConstraintFactory.ConstraintSet;
import fr.lirmm.coconut.acquisition.core.acqsolver.ACQ_ConstraintSolver;
import fr.lirmm.coconut.acquisition.core.acqsolver.ACQ_IDomain;
import fr.lirmm.coconut.acquisition.core.acqsolver.SATModel;
import fr.lirmm.coconut.acquisition.core.acqsolver.SATSolver;
import fr.lirmm.coconut.acquisition.core.learner.ACQ_Bias;
import fr.lirmm.coconut.acquisition.core.learner.ACQ_Learner;
import fr.lirmm.coconut.acquisition.core.learner.ACQ_Query;
import fr.lirmm.coconut.acquisition.core.learner.ACQ_Scope;
import fr.lirmm.coconut.acquisition.core.tools.Chrono;
import fr.lirmm.coconut.acquisition.core.tools.FileManager;
import fr.lirmm.coconut.acquisition.core.learner.*;


public class ACQ_CONACQv1 {
	protected ACQ_Bias bias;
	protected ACQ_Learner learner;
	protected ACQ_ConstraintSolver constrSolver;
	protected ACQ_IDomain domain;
	protected boolean verbose = false;
	protected boolean log_queries = false;
	public ConstraintMapping mapping;
	protected SATSolver satSolver;
	protected ConstraintFactory constraintFactory;
	protected ACQ_Network learned_network;
	protected ACQ_Network init_network;
	protected ArrayList<ACQ_Network> strategy = null;
	protected ContradictionSet backgroundKnowledge = null;
	protected Chrono chrono;
	protected int bias_size_before_preprocess;
	protected int bias_size_after_preprocess;
	protected int max_random = 0;
	protected int n_asked = 0;
	protected ArrayList<ACQ_Query> queries;
	protected ArrayList<ConstraintSet> Clauses;

	protected ACQ_Network minimalNetwork;
	protected ACQ_Network mostSpecificNetwork;

	protected ACQ_CNF T;
	protected ContradictionSet N;

	public ACQ_CONACQv1(ACQ_Learner learner, ACQ_Bias bias, SATSolver sat, ACQ_ConstraintSolver solv, String queries) {
		this.bias = bias;
		this.constraintFactory = bias.network.getFactory();
		this.learner = learner;
		this.satSolver = sat;
		this.constrSolver = solv;
		this.domain = solv.getDomain();
		this.mapping = new ConstraintMapping();
		this.queries = getQueries(queries);
		this.init_network = new ACQ_Network(constraintFactory, bias.getVars());

		for (ACQ_IConstraint c : bias.getConstraints()) {
			String newvarname = c.getName() + c.getVariables();
			Unit unit = this.satSolver.addVar(c, newvarname);
			this.mapping.add(c, unit);
		}
		assert mapping.size() == bias.getSize() : "mapping and bias must contain the same number of elements";
		filter_conjunctions();
		bias_size_before_preprocess = bias.getSize();

	}

	private ArrayList<ACQ_Query> getQueries(String queries2) {
		ArrayList<ACQ_Query> queries = new ArrayList<ACQ_Query>();
		BufferedReader reader;//
		try {
			reader = new BufferedReader(new FileReader("ConstraintAcquisition/benchmarks/queries/" + queries2 + ".queries"));
			String line;
//			String str;
			while (((line = reader.readLine()) != null)) {
				String[] lineSplited = line.split(" ");
				int[] values = new int[lineSplited.length - 1];

				int label = Integer.parseInt(lineSplited[lineSplited.length - 1]);
				int i = 0;
				for (String s : lineSplited) {
					if (i == lineSplited.length - 1)
						break;
					values[i] = Integer.parseInt(s);
					i++;
				}
				BitSet bs = new BitSet();
				bs.set(0, i);
				ACQ_Scope scope = new ACQ_Scope(bs);
				ACQ_Query q = new ACQ_Query(scope, values);
				if (label == 1)
					q.classify(true);
				else
					q.classify(false);

				queries.add(q);

			}
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return queries;
	}

	protected void filter_conjunctions() {
		for (Unit unit : mapping.values()) {
			ACQ_IConstraint c = unit.getConstraint();
			if (c instanceof ACQ_ConjunctionConstraint) {
				bias.reduce(c);
			}
		}
	}

	public void setMaxRand(int max) {
		this.max_random = max;
	}

	public void setStrat(ArrayList<ACQ_Network> strat) {
		this.strategy = strat;
	}

	public void setBackgroundKnowledge(ContradictionSet back) {
		this.backgroundKnowledge = back;
	}

	public float getPreprocessDiminution() {
		return ((float) (100 * (bias_size_before_preprocess - bias_size_after_preprocess))
				/ bias_size_before_preprocess);
	}

	public ACQ_Bias getBias() {
		return bias;
	}

	public ACQ_Network getLearnedNetwork() {
		return learned_network;
	}

	public void setVerbose(boolean verbose) {

		this.verbose = verbose;
	}

	public void setLog_queries(boolean logqueries) {

		this.log_queries = logqueries;
	}


	protected ACQ_Query getQuery(String line){
		assert (line!=null):"the sample is null";

			String[] lineSplited = line.split(" ");
			int[] values = new int[lineSplited.length - 1];

			int label = Integer.parseInt(lineSplited[lineSplited.length - 1]);
			int i = 0;
			for (String s : lineSplited) {
				if (i == lineSplited.length - 1)
					break;
				values[i] = Integer.parseInt(s);
				i++;
			}
			BitSet bs = new BitSet();
			bs.set(0, i);
			ACQ_Scope scope = new ACQ_Scope(bs);
			ACQ_Query q = new ACQ_Query(scope, values);
			if (label == 1)
				q.classify(true);
			else
				q.classify(false);

			return q;
	}


	public boolean process(Chrono chronom, int max_queries) throws Exception {
		chrono = chronom;

		max_queries = 5; // should get from method
		boolean convergence = false;
		Clauses = new ArrayList<ConstraintSet>();
		minimalNetwork = new ACQ_Network(constraintFactory, bias.getVars());
		mostSpecificNetwork = new ACQ_Network(constraintFactory, bias.getVars());
		boolean collapse = false;
		ACQ_CNF T = new ACQ_CNF();
		ContradictionSet N;
		if (this.backgroundKnowledge == null) {
			N = new ContradictionSet(constraintFactory, bias.network.getVariables(), mapping);
		} else {
			N = this.backgroundKnowledge;
		}

		// assert(learned_network.size()==0);
		chrono.start("total_acq_time");

		bias_size_after_preprocess = bias.getSize();

		ArrayList<String> asked = new ArrayList<>();

		if (!init_network.isEmpty()) {

			for (ACQ_IConstraint c : init_network.getConstraints()) {
				Unit unit = mapping.get(c).clone();
				T.unitPropagate(unit, chrono);
				N.unitPropagate(unit, chrono);
				T.add(new ACQ_Clause(mapping.get(c)));

			}
			bias.reduce(init_network.getConstraints());
		}
		// for to implement incremental process
		// 1. stop condition when Cs = Cm 2. examples used-up 3. make size with n
		//acquisition get RL answer from environment via files( .queries) or  other way.
		// RL get Cs/Cm from CARL algorithm via file (target.network) and prepare logic to
		// or get the answer from the learner , how to communicate between (e.g: zeroMessageQ), or DB.
		// n = 100 , max = 9

		// counters of batchs
		int batchs = (queries.size() / max_queries) + 1;

		Iterator<ACQ_Query> iter_queries = queries.iterator();

		System.out.println("batchs : " + String.valueOf(batchs));
		// create batch with count of  max_queries
		for (int batch_counter =0 ;batch_counter < batchs ; batch_counter++) {

			ArrayList<ACQ_Query> batch_query = new ArrayList<ACQ_Query>();

			for (int qsize = 0; qsize < max_queries; qsize++)
				if(iter_queries.hasNext())
					batch_query.add(iter_queries.next());

			for (ACQ_Query membership_query : batch_query) {

				if (bias.getConstraints().isEmpty())
					break;

				assert membership_query != null : "membership query can't be null";

				assert !asked.contains(membership_query.toString());
				asked.add(membership_query.toString());
				if (verbose)
					System.out.print(membership_query.getScope() + "::" + Arrays.toString(membership_query.values));
				boolean answer = membership_query.isPositive();
				if (log_queries)
					FileManager.printFile(membership_query, "queries");
				n_asked++;
				ConstraintSet kappa = bias.getKappa(membership_query);

				assert kappa.size() > 0;
				if (verbose)
					System.out.println("::" + membership_query.isPositive());
				if (answer) {
					for (ACQ_IConstraint c : kappa) {
						Unit unit = mapping.get(c).clone();
						unit.setNeg();
						T.unitPropagate(unit, chrono);
						N.unitPropagate(unit, chrono);
					}
					bias.reduce(kappa);

					for (ConstraintSet c : Clauses) {
						for (int i = 0; i < kappa.size(); i++) {
							if (c.contains(kappa.get_Constraint(i))) {
								c.remove(kappa.get_Constraint(i));
							}
						}
					}

				} else {
					Clauses.add(kappa);
					if (kappa.size() == 1) {
						ACQ_IConstraint c = kappa.get_Constraint(0);
						Unit unit = mapping.get(c).clone();
						T.unitPropagate(unit, chrono);
						N.unitPropagate(unit, chrono);
						T.add(new ACQ_Clause(mapping.get(c)));
						bias.reduce(c.getNegation());
					} else {
						ACQ_Clause disj = new ACQ_Clause();
						for (ACQ_IConstraint c : kappa) {
							Unit unit = mapping.get(c).clone();
							disj.add(unit);
						}
						T.add(disj);
						// T.unitPropagate(chrono);
					}
				}

			}

				//Compute the minimal Network (= constraints in unit clauses)
				for(ConstraintSet c:Clauses) {
					if(c.size()==1)
						minimalNetwork.add(c.get_Constraint(0), true);
				}
				System.out.println("############## CM (Minimal Network)##############");
				System.out.println(minimalNetwork.getConstraints().toString2());

				//Compute the most specific Network (= all constraints in the bias)
				for (ACQ_IConstraint constr: bias.getConstraints()) {
					mostSpecificNetwork.add(constr, true);
				}

				System.out.println("############## CS (Moast Spesific Network)##############");
				System.out.println(mostSpecificNetwork.getConstraints().toString2());

				//learned_network.clean();

			//comparing cs & cm if there were same means it converged
			boolean same_cs_cm = true;
			Iterator<ACQ_IConstraint> cs_iter = mostSpecificNetwork.iterator() ;
			Iterator<ACQ_IConstraint> cm_iter = minimalNetwork.iterator();

			while(cs_iter.hasNext()){
				ACQ_IConstraint cons=  cs_iter.next();
				System.out.println(cons.getName());
				if( !cm_iter.hasNext() || !cons.equals(cm_iter.next()))
					same_cs_cm= false;
			}

			//  bias  = CS
			bias =new ACQ_Bias(mostSpecificNetwork);
			if (same_cs_cm) break;
		}
		chrono.stop("total_acq_time");

		System.out.println("example : 3 3 3 1 is --> "+ classify(getQuery("3 3 3 1")));
		System.out.println("example : 4 3 2 1 is --> "+ classify(getQuery("4 3 2 1")));
		
		return !collapse;
	}

	public Classification classify(ACQ_Query query) {
//		updateNetworks();
		boolean generalAccepts = false;
		boolean isCompleteQuery = true;
		boolean specificAccepts = false;

		assert (minimalNetwork!=null || mostSpecificNetwork!=null ):"the network is not ready, please continue learning";


		if (isCompleteQuery) {
			generalAccepts = minimalNetwork.check(query);
			specificAccepts = mostSpecificNetwork.check(query);
		}
		if (generalAccepts == false && specificAccepts == false) return Classification.UNKNOWN;

		return (generalAccepts ? Classification.NEGATIVE:Classification.POSITIVE);
	}

//	protected boolean isUnset(ACQ_IConstraint constr, ACQ_CNF T, ContradictionSet N) {
//		Unit unit = mapping.get(constr);
//		Unit neg = unit.clone();
//		neg.setNeg();
//
//		ACQ_CNF tmp1 = T.clone();
//		tmp1.concat(N.toCNF());
//		ACQ_CNF tmp2 = tmp1.clone();
//
//		tmp1.add(new ACQ_Clause(unit));
//		tmp2.add(new ACQ_Clause(neg));
//
//		return satSolver.solve(tmp1) != null && satSolver.solve(tmp2) != null;
//	}

//	protected void updateNetworks() {
//		assert (T.isMonomial());
//
//		if (minimalNetwork == null) {
//			SATModel model = satSolver.solve(T);
//			minimalNetwork = new ACQ_Network(constraintFactory, bias.getVars());
//
//			for (Unit unit : mapping.values()) {
//				if (!unit.isNeg() && model.get(unit)) {
//					minimalNetwork.add(unit.getConstraint(), true);
//				}
//			}
//		}
//
//		if (mostSpecificNetwork == null) {
//			mostSpecificNetwork = new ACQ_Network(constraintFactory, bias.getVars());
//			for (ACQ_IConstraint constr: bias.getConstraints()) {
//				if (!isUnset(constr, T, N)) {
//					mostSpecificNetwork.add(constr, true);
//				}
//			}
//		}
//	}




	public ACQ_Network getMinimalNetwork() {
		return minimalNetwork;
	}

	public void setMinimalNetwork(ACQ_Network minimalNetwork) {
		this.minimalNetwork = minimalNetwork;
	}

	public ACQ_Network getMostSpecificNetwork() {
		return mostSpecificNetwork;
	}

	public void setMostSpecificNetwork(ACQ_Network mostSpecificNetwork) {
		this.mostSpecificNetwork = mostSpecificNetwork;
	}

}
