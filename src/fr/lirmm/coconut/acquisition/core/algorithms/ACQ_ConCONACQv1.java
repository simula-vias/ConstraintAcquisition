package fr.lirmm.coconut.acquisition.core.algorithms;


import fr.lirmm.coconut.acquisition.core.acqconstraint.*;
import fr.lirmm.coconut.acquisition.core.acqsolver.SATSolver;
import fr.lirmm.coconut.acquisition.core.learner.*;
import fr.lirmm.coconut.acquisition.core.tools.Chrono;
import nill.morena.services.BIOSService;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;


public class ACQ_ConCONACQv1 {
    private static ACQ_ConCONACQv1 singleton = null;

    protected ACQ_ConCONACQv1_Model[] models;

    protected boolean verbose = false;
    protected boolean log_queries = false;

    protected ConstraintFactory constraintFactory;
    protected ArrayList<ACQ_Network> strategy = null;
    protected ContradictionSet backgroundKnowledge = null;
    protected Chrono chrono;
    protected int max_random = 0;
    protected int n_asked = 0;
    protected int n_asked_positive = 0;
    protected int n_asked_negative = 0;
    protected ArrayList<ACQ_Query> queries;

    public static ACQ_ConCONACQv1 initiate(ACQ_Bias[] bias, SATSolver[] sat, Chrono chrono) {
        if (singleton != null) return null;
        // TODO We could copy the bias, but if we pass a sat solver array, then we can also put a bias array...
        singleton = new ACQ_ConCONACQv1(bias, sat, chrono);
        return singleton;
    }

    public static ACQ_ConCONACQv1 getInstance() {
        return singleton;
    }

    private ACQ_ConCONACQv1(ACQ_Bias[] bias, SATSolver[] sat, Chrono chrono) {
        assert sat.length == bias.length : "need one SATSolver per context";

        int numContexts = bias.length;

        this.chrono = chrono;
        this.models = new ACQ_ConCONACQv1_Model[numContexts];

        for (int i = 0; i < numContexts; i++) {
            String logfile = BIOSService.getBIOS().getString("CALogFile") + ".context" + i;
            this.models[i] = new ACQ_ConCONACQv1_Model(bias[i], sat[i], verbose, chrono);
            this.models[i].setLogfile(logfile);
        }

//        this.bias = bias;
//        this.satSolver = sat;
//        this.constrSolver = solv;
//        this.domain = solv.getDomain();
//
////		this.queries = getQueries(queries);
//        this.init_network = new ACQ_Network(constraintFactory, bias.getVars());
//
//        for (ACQ_IConstraint c : bias.getConstraints()) {
//            String newvarname = c.getName() + c.getVariables();
//            Unit unit = this.satSolver.addVar(c, newvarname);
//            this.mapping.add(c, unit);
//        }
//        assert mapping.size() == bias.getSize() : "mapping and bias must contain the same number of elements";
//        filter_conjunctions();
//        bias_size_before_preprocess = bias.getSize();

    }

    public static ACQ_Query getQuery(String line) {
        assert (line != null) : "the sample is null";

        String[] lineSplited = line.split(" ");
        int lengthSplit = lineSplited.length;
        int[] values = new int[lengthSplit - 1];

        int label = Integer.parseInt(lineSplited[lengthSplit - 1]);
        int i = 0;
        for (String s : lineSplited) {
            if (i == lengthSplit - 1) break;
            if (s != null && !s.trim().isEmpty()) {
                values[i] = Integer.parseInt(s);
                i++;
            }
        }
        BitSet bs = new BitSet();
        bs.set(0, i);
        ACQ_Scope scope = new ACQ_Scope(bs);
        ACQ_Query q = new ACQ_Query(scope, values);
//        System.out.println("bitSet length: " + i + "values in line: " + values.length);

        if (label != -1) {
            q.classify(label == 1);
        }

        return q;
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

    public ACQ_Bias getBias(int context) {
        if (!isValidContext(context)) return null;
        return models[context].getBias();
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    public void setLog_queries(boolean logqueries) {

        this.log_queries = logqueries;
    }

    public boolean process(Chrono chronom, int max_queries) throws Exception {
        chrono = chronom;
        ArrayList<ACQ_Query> openQueries = new ArrayList<>();

        long filePointer = 0;

        String runDirectory = BIOSService.getBIOS().getString("examplePath") + BIOSService.getBIOS().getString("file");

        File queryFile = Paths.get(runDirectory + "/queries.txt").toFile();
        System.out.println("Query file: " + queryFile.getAbsolutePath());

        RandomAccessFile file = null;

        try {
            while (!queryFile.exists()) {
                System.out.println("Query file does not (yet) exist... wait 5s");
                Thread.sleep(5000);
            }

            // Start tailing
            file = new RandomAccessFile(queryFile, "r");
            while (true) {
                // HS: One query file should be enough
                // We can read it continuously
                try {
                    // Compare the length of the file to the file pointer
                    long fileLength = queryFile.length();
                    if (fileLength < filePointer) {
                        // Log file must have been rotated or deleted;
                        // reopen the file and reset the file pointer
                        file = new RandomAccessFile(queryFile, "r");
                        filePointer = 0;
                    }

                    if (fileLength > filePointer) {
                        // There is data to read
                        file.seek(filePointer);
                        String line = file.readLine();
                        while (line != null) {
                            openQueries.add(getQuery(line)); // add to query queue

                            line = file.readLine();
                        }
                        filePointer = file.getFilePointer();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }

                while (!openQueries.isEmpty()) {
                    ACQ_Query membership_query = openQueries.remove(0);
                    assert membership_query != null : "membership query can't be null";

                    int context = getContext(membership_query);

                    System.out.println("Send query to context " + context);
                    this.models[context].learn(membership_query);

                    n_asked++;
                    if (membership_query.isPositive()) {
                        n_asked_positive++;
                    } else if (membership_query.isNegative()) {
                        n_asked_negative++;
                    }
                }

                Thread.sleep(15000);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (file != null) {
                // Close the file that we are tailing
                file.close();
            }
        }

        chrono.stop("total_acq_time");

        return true;
    }

    private static int getContext(ACQ_Query query) {
        assert  query != null : " query is null ";
        assert query.values[query.values.length - 1] != -1 :" query is -1";

        return query.values[query.values.length - 1];
    }

    private boolean isValidContext(int context) {
        assert 0 <= context && context < models.length : "context is not valid";
        return 0 <= context && context < models.length;
    }

    public Classification classify(ACQ_Query query) {
        int context = getContext(query);
        assert context != -1 : "context is -1";

        if (!isValidContext(context)) return Classification.UNKNOWN;

        return models[context].classify(query);
    }
}
