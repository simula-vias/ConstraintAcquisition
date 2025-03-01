package fr.lirmm.coconut.acquisition.core.acqconstraint;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.jgrapht.Graph;
import org.jgrapht.alg.clique.DegeneracyBronKerboschCliqueFinder;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.DefaultUndirectedGraph;

import fr.lirmm.coconut.acquisition.core.acqconstraint.ConstraintFactory.ConstraintSet;
import fr.lirmm.coconut.acquisition.core.learner.ACQ_Query;
import fr.lirmm.coconut.acquisition.core.learner.ACQ_Scope;
import fr.lirmm.coconut.acquisition.core.tools.NameService;

/**
 * Class used to represent constraint network.
 */
public class ACQ_Network implements Iterable<ACQ_IConstraint> {

	private final transient PropertyChangeSupport pcs = new PropertyChangeSupport(this);

	/**
	 * Constraint factory
	 */
	public ConstraintFactory constraintFactory;
	/**
	 * Variables of this network
	 */
	private ACQ_Scope variables;
	/**
	 * Set of constraints of this network
	 */
	private ConstraintSet constraints;

	/**
	 * Empty constructor
	 */
	public ACQ_Network(ConstraintFactory factory) {
		this(factory, null, null, true);
	}

	/**
	 * Constructor from a specified scope (variables)
	 * 
	 * @param variables Scope that contains variables
	 */
	public ACQ_Network(ConstraintFactory factory, ACQ_Scope variables) {
		this(factory, variables, null, true);
	}

	/**
	 * Constructor from a specified set of constraints
	 * 
	 * @param constraints Set of constraints
	 */
	public ACQ_Network(ConstraintFactory factory, ConstraintSet constraints) {
		// variable set is populated by used constraints
		this(factory, null, constraints, true);
	}

	/**
	 * Constructor from a specified scope and set of constraints
	 * 
	 * @param variables   Scope
	 * @param constraints Set of constraints
	 */
	public ACQ_Network(ConstraintFactory factory, ACQ_Scope variables, ConstraintSet constraints) {
		// as default constraints are only added when all variables are already
		// contained in variable set
		this(factory, variables, constraints, false);
	}

	/**
	 * Constructor from an other network and a specified scope It takes the
	 * constraints from the other network and the scope from the specified scope
	 * 
	 * @param other_network Network to take constraints from
	 * @param scope         Scope
	 */
	public ACQ_Network(ConstraintFactory factory, ACQ_Network other_network, ACQ_Scope scope) {
		this(factory, scope, other_network.constraints, false);
	}

	/**
	 * Main constructor of networks Build a network from the specified scope and the
	 * specified set of constraints
	 * 
	 * @param variables     Scope
	 * @param constraintSet   Set of constraints
	 * @param add_variables if false constraints are only added when all variables
	 *                      are already contained in variable set, else it will add
	 *                      the constraint and the missing variables
	 */
	public ACQ_Network(ConstraintFactory factory, ACQ_Scope variables, ConstraintSet constraintSet,
			boolean add_variables) {
		this.constraintFactory = factory;
		this.constraints = constraintFactory.createSet();
		if (variables == null) {
			this.variables = new ACQ_Scope();
		} else {
			this.variables = variables;
		}
		if (constraintSet != null) {
			constraintSet.forEach((x) -> {
				add(x, add_variables);
			});
		}
	}

	public ACQ_Network() {
	}

	/**
	 * Returns an iterator of the set of constraints of this network
	 * 
	 * @return iterator of the set of constraints
	 */
	@Override
	public Iterator<ACQ_IConstraint> iterator() {
		return constraints.iterator();
	}

	/**
	 * Returns the size of the set of constraints
	 * 
	 * @return amount of constraints
	 */
	public int size() {
		return constraints.size();
	}

	public boolean isEmpty() {
		return constraints == null || size() == 0;
	}

	/**
	 * Returns the set of constraints of this network
	 * 
	 * @return set of constraints
	 */
	public ConstraintSet getConstraints() {
		return this.constraints;
	}

	/**
	 * Returns the set of variables of this network
	 * 
	 * @return set of variables
	 */
	public ACQ_Scope getVariables() {
		return this.variables;
	}

	/**
	 * Add a specified constraint to the set of this network If force is set to
	 * false, the specified constraint will only be added if all the variables of
	 * the specified constraint are contained in the variable set of this network
	 * Else it will add the constraint and the missing variables
	 * 
	 * @param cst   Constraint to add
	 * @param force Determine the way of adding
	 */
	synchronized public final void add(ACQ_IConstraint cst, boolean force) {
		if (force || this.variables.containsAll(cst.getScope())) {
			this.variables = variables.union(cst.getScope());
			pcs.firePropertyChange("ADD_VARIABLES", this.variables, cst.getScope());
			this.constraints.add(cst);
			String threadName = Thread.currentThread().getName();
			threadName = threadName.replaceAll("-", "_");
			pcs.firePropertyChange("ADD_CONSTRAINT", threadName, cst);
		}
	}

	synchronized public void addAll(ConstraintSet constraintSet, boolean force) {

		for (ACQ_IConstraint cst : constraintSet)
			this.add(cst, force);
	}

	public void addAll(ACQ_Network constraintNet, boolean force) {

		for (ACQ_IConstraint cst : constraintNet.constraints)
			this.add(cst, force);
	}

	public boolean contains(ACQ_IConstraint cst) {

		for (ACQ_IConstraint cst_ : this.constraints)
			if (cst_.equals(cst))
				return true;
		return false;
	}

	/**
	 * Display all the details of the variables of this network
	 * 
	 */
	public void printVariables() {
		for (int var : variables) {
			System.out.println("var " + var + "=" + NameService.getVarName(var));
		}
	}

	/**
	 * Checks if the query violates this network of constraints
	 * 
	 * @param query Example
	 * @return false if query violates at least one constraint of this network
	 */
	public boolean check(ACQ_Query query) {
		for (ACQ_IConstraint constraint : constraints) {
			if (query.getScope().containsAll(constraint.getScope())
					&& !constraint.checker(constraint.getProjection(query))) {
				return false;
			}
		}
		return true;
	}

	/**************************
	 * checkNotAllNeg
	 * 
	 * @param query
	 * @return true if the tuple represented by query is rejected by at least a
	 *         constraint of the network
	 * 
	 * @author NADJIB
	 **************************/
	public boolean checkNotAllNeg(ACQ_Query query) {
		int counter1 = 0, counter2 = 0;
		for (ACQ_IConstraint constraint : constraints) {
			if (query.getScope().containsAll(constraint.getScope())) {
				counter1++;

				if (constraint.checker(constraint.getProjection(query)))
					counter2++;

			}

		}
		return (counter2 != 0) && (counter2 != counter1);
	}

	@Override
	public String toString() {
		String s = "network var=" + variables.size() + " cst=" + constraints.size() + "\n";

		s += "-------------------------\n";
		s += "CONSTRAINTS:\n";
		for (ACQ_IConstraint c : this.constraints)
			s += c.toString() + "\n";
		s += "-------------------------\n";

		return s;
	}

	public void removeAll() {
		while (!constraints.isEmpty())
			remove(constraints.iterator().next());
	}

	public void removeAll(ConstraintSet set) {
		constraints.removeAll(set);
		for (ACQ_IConstraint cst : set)
			pcs.firePropertyChange("REMOVE_CONSTRAINT", cst, null);
		pcs.firePropertyChange("EMPTY_NETWORK", null, null);
	}

	public void remove(ACQ_IConstraint cst) {
		this.constraints.remove(cst);
		pcs.firePropertyChange("REMOVE_CONSTRAINT", cst, null);
		if (constraints.isEmpty()) {
			pcs.firePropertyChange("EMPTY_NETWORK", cst, null);
		}
	}

	public void allDiffCliques() {

		DegeneracyBronKerboschCliqueFinder<Integer, DefaultEdge> finder = new DegeneracyBronKerboschCliqueFinder<>(
				this.buildGraph());

		// finder.lazyRun();

		for (Set<Integer> clique : finder) {

			List<Integer> list = new ArrayList<Integer>(clique);
			for (int i = 0; i < list.size(); i++)
				for (int j = i + 1; j < list.size(); j++) {
					this.remove(new BinaryArithmetic("DifferentXY", list.get(i), Operator.NEQ, list.get(j), "EqualXY"));
					this.remove(new BinaryArithmetic("DifferentXY", list.get(j), Operator.NEQ, list.get(i), "EqualXY"));
				}
			this.add(new AllDiff(clique), true);
		}
	}

	private Graph<Integer, DefaultEdge> buildGraph() {

		Graph<Integer, DefaultEdge> graph = new DefaultUndirectedGraph<>(DefaultEdge.class);

		for (ACQ_IConstraint c : this) {
			if (c.getName().equals("DifferentXY")) {
				graph.addVertex(c.getVariables()[0]);
				graph.addVertex(c.getVariables()[1]);
				graph.addEdge(c.getVariables()[0], c.getVariables()[1]);
			}
		}

		return graph;

	}

	public ACQ_Network getProjection(ACQ_Scope scope) {

		ACQ_Network result = new ACQ_Network(constraintFactory, scope);

		for (ACQ_IConstraint cst : this.constraints) {
			if (scope.containsAll(cst.getScope()))
				result.add(cst, true);

		}
		return result;
	}

	public ACQ_Network getExactProjection(ACQ_Scope scope) {

		ACQ_Network result = new ACQ_Network(constraintFactory, scope);

		for (ACQ_IConstraint cst : this.constraints) {
			if (scope.equals(cst.getScope()))
				result.add(cst, true);

		}
		return result;
	}

	public ConstraintFactory getFactory() {
		return constraintFactory;
	}

	public void addPropertyChangeListener(PropertyChangeListener l) {
		pcs.addPropertyChangeListener(l);
		l.propertyChange(new PropertyChangeEvent(this, "INIT_GRAPH", this, this));
	}

	public void removePropertyChangeListener(PropertyChangeListener l) {
		pcs.removePropertyChangeListener(l);
	}

	public ACQ_IConstraint[] getArrayConstraints() {
		ACQ_IConstraint[] csts = new ACQ_IConstraint[this.constraints.size()];
		ArrayList<ACQ_IConstraint> cstsList = new ArrayList<>(this.constraints.size());
		this.constraints.iterator().forEachRemaining(cstsList::add);
		return cstsList.toArray(csts);
	}

	public void clean() {
		for (int i = size() - 1; i >= 0; i--) {
			ACQ_IConstraint constr = getConstraints().get_Constraint(i);
			if (constr instanceof ACQ_DisjunctionConstraint) {
				for (int j = 0; j < i; j++) {
					if (((ACQ_DisjunctionConstraint) constr).contains(getConstraints().get_Constraint(j))) {
						this.remove(constr);
						break;
					}
				}
			} else if (constr instanceof ACQ_ConjunctionConstraint) {
				this.remove(constr); // We don't need ConjunctionConstraints as we can express it as conjunction of
										// Constraints
			}
		}
	}

}
