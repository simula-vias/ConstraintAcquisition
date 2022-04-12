package fr.lirmm.coconut.acquisition.core.algorithms;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;

import fr.lirmm.coconut.acquisition.core.acqconstraint.ACQ_CNF;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ACQ_Clause;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ACQ_ConjunctionConstraint;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ACQ_Formula;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ACQ_IConstraint;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ACQ_Network;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ConstraintFactory;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ConstraintMapping;
import fr.lirmm.coconut.acquisition.core.acqconstraint.Contradiction;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ContradictionSet;
import fr.lirmm.coconut.acquisition.core.acqconstraint.Unit;
import fr.lirmm.coconut.acquisition.core.acqconstraint.ConstraintFactory.ConstraintSet;
import fr.lirmm.coconut.acquisition.core.acqsolver.ACQ_ConstraintSolver;
import fr.lirmm.coconut.acquisition.core.acqsolver.ACQ_IDomain;
import fr.lirmm.coconut.acquisition.core.acqsolver.SATModel;
import fr.lirmm.coconut.acquisition.core.acqsolver.SATSolver;
import fr.lirmm.coconut.acquisition.core.learner.*;
import fr.lirmm.coconut.acquisition.core.tools.Chrono;
import fr.lirmm.coconut.acquisition.core.tools.FileManager;

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
	protected ACQ_CNF T;
	protected ContradictionSet N;

	protected ACQ_Network mostGeneralNetwork = null;
	protected ACQ_Network mostSpecificNetwork = null;
	
	public ACQ_CONACQv1(ACQ_Learner learner, ACQ_Bias bias, SATSolver sat, ACQ_ConstraintSolver solv,String queries) {
		this.bias = bias;
		this.constraintFactory = bias.network.getFactory();
		this.learner = learner;
		this.satSolver = sat;
		this.constrSolver = solv;
		this.domain = solv.getDomain();
		this.mapping = new ConstraintMapping();
		this.queries=getQueries(queries);
		this.init_network= new ACQ_Network(constraintFactory, bias.getVars());
		
		for (ACQ_IConstraint c : bias.getConstraints()) {
			String newvarname = c.getName() + c.getVariables();
			Unit unit = this.satSolver.addVar(c, newvarname);
			this.mapping.add(c, unit);
		}
		assert mapping.size() == bias.getSize(): "mapping and bias must contain the same number of elements";
		filter_conjunctions();
		bias_size_before_preprocess = bias.getSize();
		
	}

	private ArrayList<ACQ_Query> getQueries(String queries2) {
		ArrayList<ACQ_Query> queries = new ArrayList<ACQ_Query>();
		BufferedReader reader;//
		try {
			reader = new BufferedReader(new FileReader("benchmarks/queries/"+queries2+".queries"));
			String line;
			String str;
			while (((line = reader.readLine()) != null)) {
				String[] lineSplited = line.split(" ");
				int [] values = new int[lineSplited.length-1];
				
				int label =  Integer.parseInt(lineSplited[lineSplited.length-1]);
				int i=0;
				for(String s : lineSplited) {
					if(i==lineSplited.length-1)
						break;
					values[i]=Integer.parseInt(s);
					i++;
				}
				BitSet bs = new BitSet();
				bs.set(0, i);
				ACQ_Scope scope = new ACQ_Scope(bs);
				ACQ_Query q= new ACQ_Query(scope,values);
				if(label == 1)
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
		return ((float) (100*(bias_size_before_preprocess - bias_size_after_preprocess)) / bias_size_before_preprocess);
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

	protected ACQ_Formula BuildFormula(Boolean splittable, ACQ_CNF T, ContradictionSet N, ACQ_Clause alpha, int t, int epsilon) {
		ACQ_Formula res =  new ACQ_Formula();
		if (!alpha.isEmpty()) {
			res.addCnf(T);
			// No need to remove unary negative as it is never added to T
			for(ACQ_IConstraint c : bias.getConstraints()) {	
				// No need to check if T contains unary negative as it is never added to T 
				boolean cont = alpha.contains(c);
				if (splittable && !cont && !alpha.contains(c.getNegation())) {
					res.addClause(new ACQ_Clause(mapping.get(c)));
				}
				if (cont) {
					ACQ_Clause newcl = new ACQ_Clause();
					newcl.add(mapping.get(c));
					newcl.add(mapping.get(c.getNegation()));
					res.addClause(newcl);
				}
			}
			
			int lower, upper;
			if (splittable) {
				lower = Math.max(alpha.getSize() - t - epsilon, 1);
				upper = Math.min(alpha.getSize() - t + epsilon, alpha.getSize() -1);
			}
			else {
				lower = 1;
				upper = alpha.getSize() - 1;
			}
			
			res.setAtLeastAtMost(alpha, lower, upper); // atLeast and atMost are left symbolic in order to let the solver encode it at will
		}
		else {
			ACQ_CNF F = T.clone();
			ACQ_Clause toadd = new ACQ_Clause();
			for (ACQ_IConstraint constr : bias.getConstraints()) {
				if (isUnset(constr, T, N)) {
					//constr is unset
					Unit toremove = mapping.get(constr.getNegation()).clone();
					toremove.setNeg();
					
					F.removeIfExists(new ACQ_Clause(toremove));
					//F.remove(new Clause(toremove));
					
					toadd.add(mapping.get(constr.getNegation()));	
				}
			}
			
			assert !toadd.isEmpty() : "toadd should not be empty";
			F.add(toadd);
			res.addCnf(F);
		}
		return res;
	}
	
	protected boolean isUnset(ACQ_IConstraint constr, ACQ_CNF T, ContradictionSet N) {
		Unit unit = mapping.get(constr);
		Unit neg = unit.clone();
		neg.setNeg();
		
		ACQ_CNF tmp1 = T.clone();
		tmp1.concat(N.toCNF());
		ACQ_CNF tmp2 = tmp1.clone();
		
		tmp1.add(new ACQ_Clause(unit));
		tmp2.add(new ACQ_Clause(neg));
		
		return satSolver.solve(tmp1) != null && satSolver.solve(tmp2) != null;
	}

	protected Boolean isConsistent(ACQ_Network network) {
		return constrSolver.solve(network);
	}
	

	public boolean process(Chrono chronom, int max_queries) throws Exception {
		chrono = chronom;
		boolean convergence = false;
		boolean collapse = false;
		T = new ACQ_CNF();

		if (this.backgroundKnowledge == null) {
			N = new ContradictionSet(constraintFactory, bias.network.getVariables(), mapping);
		} else {
			N = this.backgroundKnowledge;
		}
		
		// assert(learned_network.size()==0);
		chrono.start("total_acq_time");
		
		//collapse = preprocess(T, N, max_queries);
		bias_size_after_preprocess = bias.getSize();
		
		ArrayList<String> asked = new ArrayList<>();
		
		if(!init_network.isEmpty()) {
			
			for (ACQ_IConstraint c : init_network.getConstraints()) {
				Unit unit = mapping.get(c).clone();
				T.unitPropagate(unit, chrono);
				N.unitPropagate(unit, chrono);
				T.add(new ACQ_Clause(mapping.get(c)));

			}
			bias.reduce(init_network.getConstraints());
		}
		
		for(ACQ_Query membership_query : queries){

			if (bias.getConstraints().isEmpty())
				break;

			assert membership_query != null : "membership query can't be null";
			
			

				assert !asked.contains(membership_query.toString());
				asked.add(membership_query.toString());
				if (verbose) System.out.print(membership_query.getScope() +"::"+ Arrays.toString(membership_query.values));
				boolean answer = membership_query.isPositive();
				if(log_queries)
					 FileManager.printFile(membership_query, "queries");
				n_asked++;
				ConstraintSet kappa = bias.getKappa(membership_query);
				assert kappa.size() > 0;
				if (verbose) System.out.println("::" + membership_query.isPositive());
				if(answer) {
					for (ACQ_IConstraint c : kappa) {
						Unit unit = mapping.get(c).clone();
						unit.setNeg();
						T.unitPropagate(unit, chrono);
						N.unitPropagate(unit, chrono);
					}
					bias.reduce(kappa);
				}
				else {
					if (kappa.size() == 1) {
						ACQ_IConstraint c = kappa.get_Constraint(0);
						Unit unit = mapping.get(c).clone();
						T.unitPropagate(unit, chrono);
						N.unitPropagate(unit, chrono);
						T.add(new ACQ_Clause(mapping.get(c)));
						bias.reduce(c.getNegation());
					}
					else {
						ACQ_Clause disj = new ACQ_Clause();
						for (ACQ_IConstraint c: kappa) {
							Unit unit = mapping.get(c).clone();
							disj.add(unit);
						}
						T.add(disj);
						//T.unitPropagate(chrono);
					}
				}
			
		}
		chrono.stop("total_acq_time");
		if (!collapse) {
			if (verbose) System.out.print("[INFO] Extract network from T: ");
			if (max_queries != n_asked) {
				SATModel model = satSolver.solve(T);
				if (verbose) System.out.println("Done");
				learned_network = toNetwork(model);
				System.out.println(learned_network);
			}
			else {
				learned_network = new ACQ_Network(constraintFactory, bias.getVars());
				for (ACQ_IConstraint constr: bias.getConstraints()) {
					if (!isUnset(constr, T, N)) {
						learned_network.add(constr, true);
					}
					
				}
				if (verbose) System.out.println("Done");
			}
			learned_network.clean();
		}

		return !collapse;
	}

	public Classification classify(ACQ_Query query) {
		updateNetworks();

		boolean isCompleteQuery = true;

		if (isCompleteQuery) {
			boolean generalAccepts = mostGeneralNetwork.check(query);
			boolean specificAccepts = mostSpecificNetwork.check(query);
		}
		return Classification.UNKNOWN;
	}

	protected void updateNetworks() {
		assert (T.isMonomial());

		if (mostGeneralNetwork == null) {
			SATModel model = satSolver.solve(T);
			mostGeneralNetwork = new ACQ_Network(constraintFactory, bias.getVars());

			for (Unit unit : mapping.values()) {
				if (!unit.isNeg() && model.get(unit)) {
					mostGeneralNetwork.add(unit.getConstraint(), true);
				}
			}
		}

		if (mostSpecificNetwork == null) {
			mostSpecificNetwork = new ACQ_Network(constraintFactory, bias.getVars());
			for (ACQ_IConstraint constr: bias.getConstraints()) {
				if (!isUnset(constr, T, N)) {
					mostSpecificNetwork.add(constr, true);
				}
			}
		}
	}

	
	protected ACQ_Network toNetwork(SATModel model) throws Exception {
		chrono.start("to_network");
		assert(model != null);
		ACQ_Network network = new ACQ_Network(constraintFactory, bias.getVars());
		
		for (Unit unit : mapping.values()) {
			if(model.get(unit)) {
				network.add(unit.getConstraint(), true);
			}
		}
		chrono.stop("to_network");
		return network;
	}

	public void setInitN(ACQ_Network InitNetwork) {
		init_network=InitNetwork;
	}

	protected ACQ_Query initializeQuery() {
		// HS Initialize an query with full scope and placeholder values
		ACQ_Scope vars = bias.getVars();
		int[] placeholderValues = new int[vars.size()];
		return new ACQ_Query(vars, placeholderValues);
	}
}
