//package fr.lirmm.coconut.acquisition.expe_conacq;
//
//
//import fr.lirmm.coconut.acquisition.core.learner.Classification;
//import fr.lirmm.coconut.acquisition.core.ACQ_Utils;
//import fr.lirmm.coconut.acquisition.core.DefaultExperienceConacq;
//import fr.lirmm.coconut.acquisition.core.acqconstraint.*;
//import fr.lirmm.coconut.acquisition.core.acqsolver.*;
//import fr.lirmm.coconut.acquisition.core.combinatorial.AllPermutationIterator;
//import fr.lirmm.coconut.acquisition.core.combinatorial.CombinationIterator;
//import fr.lirmm.coconut.acquisition.core.learner.*;
//import org.json.JSONArray;
//import org.json.JSONObject;
//import org.zeromq.SocketType;
//import org.zeromq.ZContext;
//import org.zeromq.ZMQ;
//
//import java.sql.*;
//import java.util.ArrayList;
//import java.util.BitSet;
//import java.util.stream.IntStream;
//
//
//public class ExpeCgrlGridworldServer extends DefaultExperienceConacq {
//    static int minDomain = 0;
//    static int maxDomain = 6 * 11 * 3;  // colors * objects * states
//    static int observableCells = 49;
//    static int inventorySize = 16;
//    static int numActions = 4;
//    static int NB_VARIABLE = observableCells + inventorySize + 1 + 1;
//    static int[] cellVars = IntStream.range(0, observableCells).toArray();
//    static int[] inventoryVars = IntStream.range(observableCells, observableCells + inventorySize).toArray();
//    static int directionVar = 65;
//    static int actionVar = 66;
//
//    // TODO Make configurable
//    private boolean useMessageQueue = true;
//    private String sqliteDbPath = "/home/helge/Sandbox/cprl/minigrid.db";
//
//    // TODO: Filter for duplicate observations / in-memory sqlite?
//
//    @Override
//    public void process() {
//        ACQ_Utils.executeConacqExperience(this);
//    }
//
//    @Override
//    public ACQ_ConstraintSolver createSolver() {
//        return new ACQ_ChocoSolver(new ACQ_IDomain() {
//            @Override
//            public int getMin(int numvar) {
//                return minDomain;
//            }
//
//            @Override
//            public int getMax(int numvar) {
//                if (numvar == actionVar) {
//                    return numActions;
//                } else if (numvar == directionVar) {
//                    return 4; // num directions
//                } else {
//                    return maxDomain;
//                }
//            }
//        }, vrs, vls);
//    }
//
//    @Override
//    public SATSolver createSATSolver() {
//        //return new Z3SATSolver();
//        return new MiniSatSolver();
//    }
//
//    @Override
//    public ArrayList<ACQ_Network> createStrategy(ACQ_Bias bias) {
//        return null;
//    }
//
//    @Override
//    public ACQ_Learner createLearner() {
//        if (useMessageQueue) {
//            return new GridworldLearnerMQ();
//        } else {
//            return new GridworldLearnerSqlite().setDbConnection(sqliteDbPath);
//        }
//    }
//
//    @Override
//    public ACQ_Bias createBias() {
//        // build All variables set
//        BitSet bs = new BitSet();
//        bs.set(0, NB_VARIABLE);
//        ACQ_Scope allVarSet = new ACQ_Scope(bs);
//        ConstraintFactory constraintFactory = new ConstraintFactory();
//        ConstraintFactory.ConstraintSet constraints = constraintFactory.createSet();
//
//        // Binary Constraints over all variables
//        CombinationIterator iterator = new CombinationIterator(NB_VARIABLE, 2);
//        while (iterator.hasNext()) {
//            int[] vars = iterator.next();
//            AllPermutationIterator pIterator = new AllPermutationIterator(2);
//            while (pIterator.hasNext()) {
//                int[] pos = pIterator.next();
//                if (vars[pos[0]] != vars[pos[1]]) {
//                    constraints.add(new BinaryArithmetic("LessOrEqualXY", vars[pos[0]], Operator.LE,
//                            vars[pos[1]], Operator.NONE, -1, "GreaterXY"));
//
//                    constraints.add(new BinaryArithmetic("LessXY", vars[pos[0]], Operator.LT,
//                            vars[pos[1]], Operator.NONE, -1, "GreaterEqualXY"));
//
//                    constraints.add(new BinaryArithmetic("GreaterXY", vars[pos[0]], Operator.GT,
//                            vars[pos[1]], Operator.NONE, -1, "LessOrEqualXY"));
//
//                    constraints.add(new BinaryArithmetic("GreaterEqualXY", vars[pos[0]], Operator.GE,
//                            vars[pos[1]], Operator.NONE, -1, "LessXY"));
//
//                    constraints.add(new BinaryArithmetic("EqualXY", vars[pos[0]], Operator.EQ,
//                            vars[pos[1]], Operator.NONE, -1, "NonEqualXY"));
//
//                    constraints.add(new BinaryArithmetic("NonEqualXY", vars[pos[0]], Operator.NEQ,
//                            vars[pos[1]], Operator.NONE, -1, "EqualXY"));
//                }
//            }
//        }
//
//        // Inventory constraints
//        for (int idx : inventoryVars) {
//            constraints.add(new UnaryArithmetic("SlotFilled", idx, Operator.GT, 0));
//        }
//
//        for (int cellIdx : cellVars) {
//            ConstraintFactory.ConstraintSet inventoryConstraints = constraintFactory.createSet();
//
//            for (int invIdx : inventoryVars) {
//                inventoryConstraints.add(new ACQ_ConjunctionConstraint(constraintFactory,
//                        new UnaryArithmetic("SlotFilled", invIdx, Operator.GT, 0),
//                        new BinaryArithmetic("CellEqInv", cellIdx, Operator.EQ, invIdx, Operator.NONE, -1, "CellNonEqInv")));
//            }
//
//            constraints.add(new ACQ_DisjunctionConstraint(constraintFactory,
//                    inventoryConstraints));
//        }
//
//        ACQ_Network network = new ACQ_Network(constraintFactory, allVarSet, constraints);
//
//        return new ACQ_Bias(network);
//    }
//
//    private class GridworldLearnerMQ extends ACQ_Learner {
//        private ZContext mqContext;
//        private ZMQ.Socket socket;
//
//        public GridworldLearnerMQ() {
//            mqContext = new ZContext();
//            socket = mqContext.createSocket(SocketType.REP);
//            socket.bind("tcp://localhost:33154");
//        }
//
//        @Override
//        public boolean ask(ACQ_Query query) {
////            if (pipe == null || pipe.)
//
//            String msg = socket.recvStr();
//
//            JSONObject obj = new JSONObject(msg);
//            JSONArray jsonVars = obj.getJSONArray("variables");
//            int[] vars = new int[jsonVars.length()];
//
//            assert query.getScope().size() == vars.length;
//
//            int idx = 0;
//            for (int numvar : query.getScope()) {
//                query.values[numvar] = jsonVars.getInt(idx);
//                idx += 1;
//            }
//
//            if (obj.has("label")) {
//                query.classify(obj.getBoolean("label"));
//            }  // otherwise this is not a training example
//
//            return query.isClassified();
//        }
//
//        @Override
//        public void answer(Classification result) {
//            JSONObject msg = new JSONObject();
//            msg.put("result", result.toString());
//            String resultString = msg.toString();
//            socket.send(resultString);
//        }
//    }
//
//    private class GridworldLearnerSqlite extends ACQ_Learner {
//        int next_rowid = 1;
//        Connection db;
//
//        private ACQ_Learner setDbConnection(String dbPath) {
//            try {
//                this.db = DriverManager.getConnection("jdbc:sqlite:" + dbPath);
//
//            } catch (SQLException ex) {
//                System.out.println("SQL Exception: " + ex.getMessage());
//            }
//            return this;
//        }
//
//        @Override
//        public boolean ask(ACQ_Query e) {
//            String query = "SELECT " +
//                    "json_extract(data, '$.variables') AS variables, " +
//                    "json_extract(data, '$.reward') AS reward " +
//                    String.format("FROM logs WHERE rowid = %d;", next_rowid);
//
//            try {
//                Statement stmt = this.db.createStatement();
//                ResultSet rs = stmt.executeQuery(query);
//
//                if (rs.next()) {
//                    JSONArray jsonVars = new JSONArray(rs.getString("variables"));
//                    int[] vars = new int[jsonVars.length()];
//
//                    assert e.getScope().size() == vars.length;
//
//                    int idx = 0;
//                    for (int numvar : e.getScope()) {
//                        e.values[numvar] = jsonVars.getInt(idx);
//                        idx += 1;
//                    }
//
//                    e.classify(rs.getInt("reward") < 0);
//
//                } else {
//                    // TODO Should not occur
//                    e.classify(false);
//                    return false;
//                }
//            } catch (SQLException ex) {
//                System.out.println("SQL Exception: " + ex.getMessage());
//            }
//
//            next_rowid += 1;
//
//            return e.isClassified();
//        }
//
//        @Override
//        public void answer(Classification result) {
//            JSONObject msg = new JSONObject();
//            msg.put("result", result.toString());
//            String resultString = msg.toString();
//            //socket.send(resultString);
//            // TODO Implement
//        }
//    }
//
//    @Override
//    public ArrayList<ACQ_Bias> createDistBias() {
//        return null;
//    }
//
//    @Override
//    public ACQ_Learner createDistLearner(int id) {
//        return null;
//    }
//
//    @Override
//    public ACQ_Network createTargetNetwork() {
//        return null;
//    }
//}
