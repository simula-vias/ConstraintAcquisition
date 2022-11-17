package fr.lirmm.coconut.acquisition.core.algorithms;

import fr.lirmm.coconut.acquisition.core.acqconstraint.*;
import fr.lirmm.coconut.acquisition.core.acqsolver.SATSolver;
import fr.lirmm.coconut.acquisition.core.learner.ACQ_Bias;
import fr.lirmm.coconut.acquisition.core.learner.ACQ_Query;
import fr.lirmm.coconut.acquisition.core.learner.Classification;
import fr.lirmm.coconut.acquisition.core.tools.Chrono;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class ACQ_ConCONACQv1_Model {
    protected ACQ_Bias bias;
    protected boolean verbose;
    protected Chrono chrono;

    protected ArrayList<ConstraintFactory.ConstraintSet> Clauses = new ArrayList<>();
    protected ConstraintMapping mapping = new ConstraintMapping();
    protected ConstraintFactory constraintFactory;
    protected SATSolver satSolver;

    protected ACQ_CNF T = new ACQ_CNF();
    protected ContradictionSet N;

    protected Set<String> asked = new HashSet<>();
    protected int n_asked = 0;
    protected int n_asked_negative = 0;
    protected int n_asked_positive = 0;

    protected ACQ_Network init_network;
    protected ACQ_Network minimalNetwork;
    protected ACQ_Network mostSpecificNetwork;

    private String logfile = null;


    public ACQ_ConCONACQv1_Model(ACQ_Bias bias, SATSolver sat, boolean verbose, Chrono chrono) {
        this.verbose = verbose;
        this.chrono = chrono;
        this.bias = bias;
        this.constraintFactory = bias.network.getFactory();
        this.satSolver = sat;

        this.init_network = new ACQ_Network(constraintFactory, bias.getVars());

        // TODO This loop takes the most time during initialization
        // If all biases are the same, we could do it only once and save time.
        for (ACQ_IConstraint c : bias.getConstraints()) {
            String newvarname = c.getName() + c.getVariables();
            Unit unit = this.satSolver.addVar(c, newvarname);
            this.mapping.add(c, unit);
        }
        assert mapping.size() == bias.getSize() : "mapping and bias must contain the same number of elements";
        filter_conjunctions();

        minimalNetwork = new ACQ_Network(constraintFactory, bias.getVars());
        mostSpecificNetwork = new ACQ_Network(constraintFactory, bias.getVars());

        // initialize with default, can be overriden by `setBackgroundKnowledge`
        N = new ContradictionSet(constraintFactory, bias.network.getVariables(), mapping);

        if (!init_network.isEmpty()) {

            for (ACQ_IConstraint c : init_network.getConstraints()) {
                Unit unit = mapping.get(c).clone();
                T.unitPropagate(unit, chrono);
                N.unitPropagate(unit, chrono);
                T.add(new ACQ_Clause(mapping.get(c)));

            }
            bias.reduce(init_network.getConstraints());
        }
    }

    public boolean learn(ACQ_Query membership_query) {
        if (bias.getConstraints().isEmpty())
            return false;

        assert !asked.contains(membership_query.toString());
        asked.add(membership_query.toString());
        if (verbose)
            System.out.println(membership_query.getScope() + "::" + Arrays.toString(membership_query.values));
        boolean answer = membership_query.isPositive();
        n_asked++;

        if (answer) {
            n_asked_positive += 1;
        } else if(membership_query.isNegative()) {
            n_asked_negative += 1;
        }

        ConstraintFactory.ConstraintSet kappa = bias.getKappa(membership_query);

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

            for (ConstraintFactory.ConstraintSet c : Clauses) {
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

        updateNetworks();
        return true;
    }

    protected boolean updateNetworks() {
        //Compute the minimal Network (= constraints in unit clauses)
        for (ConstraintFactory.ConstraintSet c : Clauses) {
            if (c.size() == 1)
                minimalNetwork.add(c.get_Constraint(0), true);
        }
        System.out.println("time: " + System.currentTimeMillis());
        System.out.println("############## CM (Minimal Network)##############");
        System.out.println("network var=" + minimalNetwork.getConstraints().getVariables().length + " cst=" + minimalNetwork.getConstraints().size());
        System.out.println(minimalNetwork.getConstraints().toString2());

        //Compute the most specific Network (= all constraints in the bias)
        for (ACQ_IConstraint constr : bias.getConstraints()) {
            mostSpecificNetwork.add(constr, true);
        }

        System.out.println("############## CS (Most Specific Network)##############");
        System.out.println("network var=" + mostSpecificNetwork.getConstraints().getVariables().length + " cst=" + mostSpecificNetwork.getConstraints().size());
//                    System.out.println(mostSpecificNetwork.getConstraints().toString2());

        //learned_network.clean();

        //comparing cs & cm if there were same means it converged
        boolean same_cs_cm = true;
        Iterator<ACQ_IConstraint> cs_iter = mostSpecificNetwork.iterator();
        Iterator<ACQ_IConstraint> cm_iter = minimalNetwork.iterator();

        while (cs_iter.hasNext()) {
            ACQ_IConstraint cons = cs_iter.next();
//                        System.out.println(cons.getName());
            if (!cm_iter.hasNext() || !cons.equals(cm_iter.next()))
                same_cs_cm = false;
        }

        //  bias  = CS
        bias = new ACQ_Bias(mostSpecificNetwork);
        if (same_cs_cm) return false;

        String logmsg = "#queries: " + n_asked;
        logmsg += " (pos: " + n_asked_positive + ", neg: " + n_asked_negative + ")";
        logmsg += " |bias|: " + bias.getSize();
        logmsg += " |CM size|: " + minimalNetwork.size() + " ";
        logmsg += " |CS size|: " + mostSpecificNetwork.size() + "\n";
        System.out.println(logmsg);
        PrintWriter a = null;
        if (this.logfile != null) {
            try {
                a =   new PrintWriter(new FileWriter(this.logfile,true));
                a.write(logmsg);
            } catch (IOException e) {
                e.printStackTrace();
                // do nothing
            }finally {
                a.close();
            }
        }

        return true;
    }

    protected void filter_conjunctions() {
        for (Unit unit : mapping.values()) {
            ACQ_IConstraint c = unit.getConstraint();
            if (c instanceof ACQ_ConjunctionConstraint) {
                bias.reduce(c);
            }
        }
    }

    public Classification classify(ACQ_Query query) {
        assert (minimalNetwork != null || mostSpecificNetwork != null) : "the network is not ready, please continue learning..";

        if (minimalNetwork.getArrayConstraints().length > 0) {
            boolean cmn = minimalNetwork.check(query);
            if (!cmn) return Classification.NEGATIVE;
        }

        if (mostSpecificNetwork.getArrayConstraints().length > 0) {
            boolean csn = mostSpecificNetwork.check(query);
            if (csn) return Classification.POSITIVE;
        }

        return Classification.UNKNOWN;
    }

    public ACQ_Bias getBias() {
        return bias;
    }

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

//    public void setBackgroundKnowledge(ContradictionSet backgroundKnowledge) {
//        N = backgroundKnowledge;
//    }

    public void setLogfile(String logfile) {
        this.logfile = logfile;
    }
}
