# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood()  # food available from current state
        newFood = successorGameState.getFood()  # food available from successor state (excludes food@successor)
        currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
        newCapsules = successorGameState.getCapsules()  # capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        score += sum([1.0 / manhattanDistance(newPos, position) for position in newCapsules])
        score += sum([1.0 / manhattanDistance(newPos, position) for position in newFood.asList()])
        score += (currentFood.count() - newFood.count())
        totaltime = sum([time for time in newScaredTimes])
        score += totaltime
        for ghostState in newGhostStates:
            if manhattanDistance(newPos, ghostState.getPosition()) <= 1:
                if totaltime > 0:
                    score += 1000
                else:
                    score -= 1000
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax_decision(gameState)
        # util.raiseNotDefined()

    def minimax_decision(self, gameState):
        max_score, action = self.max_value(gameState, self.index, 0)
        return action

    def max_value(self, gameState, index, depth):
        scores = []
        legalMoves = gameState.getLegalActions(index)
        # if 'Stop' in legalMoves:
        #     legalMoves.remove('Stop')
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            return self.evaluationFunction(gameState), None
        for action in legalMoves:
            state = gameState.generateSuccessor(index, action)
            scores.append(self.min_value(state, 1, depth)[0])
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return bestScore, legalMoves[chosenIndex]

    def min_value(self, gameState, index, depth):
        scores = []
        minMaxFlag = index >= gameState.getNumAgents() - 1
        legalMoves = gameState.getLegalActions(index)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None
        for action in legalMoves:
            state = gameState.generateSuccessor(index, action)
            if minMaxFlag:
                scores.append(self.max_value(state, self.index, depth + 1)[0])
            else:
                scores.append(self.min_value(state, index + 1, depth)[0])
        bestScore = min(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return bestScore, legalMoves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax_decision(gameState)
        # util.raiseNotDefined()

    def minimax_decision(self, gameState):
        max_score, action = self.max_value(gameState, self.index, 0, float('-inf'), float('inf'))
        # print max_score
        return action

    def max_value(self, gameState, index, depth, alpha, beta):
        v = float('-inf')
        move = None
        legalMoves = gameState.getLegalActions(index)
        # if 'Stop' in legalMoves:
        #     legalMoves.remove('Stop')
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            return self.evaluationFunction(gameState), None
        for action in legalMoves:
            state = gameState.generateSuccessor(index, action)
            value = self.min_value(state, 1, depth, alpha, beta)[0]
            if v < value:
                v = value
                move = action
            if v > beta:
                return v, move
            alpha = max(alpha, v)
        return v, move

    def min_value(self, gameState, index, depth, alpha, beta):
        v = float('inf')
        move = None
        minMaxFlag = index >= gameState.getNumAgents() - 1
        legalMoves = gameState.getLegalActions(index)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None
        for action in legalMoves:
            state = gameState.generateSuccessor(index, action)
            if minMaxFlag:
                value = self.max_value(state, self.index, depth + 1, alpha, beta)[0]
            else:
                value = self.min_value(state, index + 1, depth, alpha, beta)[0]
            if v > value:
                v = value
                move = action
            if v < alpha:
                return v, move
            beta = min(beta, v)
        return v, move


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.minimax_decision(gameState)
        # util.raiseNotDefined()

    def minimax_decision(self, gameState):
        max_score, action = self.max_value(gameState, self.index, 0)
        return action

    def max_value(self, gameState, index, depth):
        scores = []
        legalMoves = gameState.getLegalActions(index)
        # if 'Stop' in legalMoves:
        #     legalMoves.remove('Stop')
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            return self.evaluationFunction(gameState), None
        for action in legalMoves:
            state = gameState.generateSuccessor(index, action)
            scores.append(self.expect_value(state, 1, depth))
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return bestScore, legalMoves[chosenIndex]

    def expect_value(self, gameState, index, depth):
        scores = []
        minMaxFlag = index >= gameState.getNumAgents() - 1
        legalMoves = gameState.getLegalActions(index)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        weight = 1.0 / len(legalMoves)
        for action in legalMoves:
            state = gameState.generateSuccessor(index, action)
            if minMaxFlag:
                scores.append(self.max_value(state, self.index, depth + 1)[0])
            else:
                scores.append(weight * self.expect_value(state, index + 1, depth))
        bestScore = sum(scores)
        return bestScore


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      Analysis:
      A. Important features:
        1. distance between pacman's position and remain foods' positions
        2. distance between pacman's position and remain capsules' positions
        3. number of remain foods
        4. distance between pacman's position and ghosts' positions
        5. vailable scared times
        6. currentGameState.getScore()

      B. Strategy:
        Use linear combination of important features as evaluation function
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()  # food available from current state
    currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    score = currentGameState.getScore()
    score += sum([1.0 / manhattanDistance(currentPos, position) for position in currentCapsules])
    score += sum([1.0 / manhattanDistance(currentPos, position) for position in currentFood.asList()])
    totaltime = sum([time for time in currentScaredTimes])
    score += totaltime
    for ghostState in currentGhostStates:
        if manhattanDistance(currentPos, ghostState.getPosition()) <= 1:
            if totaltime > 0:
                score += 1000
            else:
                score -= 1000
    return score


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.minimax_decision(gameState)

    def minimax_decision(self, gameState):
        max_score, action = self.max_value(gameState, self.index, 0, float('-inf'), float('inf'))
        # print max_score
        return action


    def max_value(self, gameState, index, depth, alpha, beta):
        v = float('-inf')
        move = None
        legalMoves = gameState.getLegalActions(index)
        # if 'Stop' in legalMoves:
        #     legalMoves.remove('Stop')
        if gameState.isLose() or gameState.isWin() or self.depth == depth:
            return contestEvaluationFunction(gameState), None
        for action in legalMoves:
            state = gameState.generateSuccessor(index, action)
            value = self.min_value(state, 1, depth, alpha, beta)[0]
            if v < value:
                v = value
                move = action
            if v > beta:
                return v, move
            alpha = max(alpha, v)
        return v, move

    def min_value(self, gameState, index, depth, alpha, beta):
        v = float('inf')
        move = None
        minMaxFlag = index >= gameState.getNumAgents() - 1
        legalMoves = gameState.getLegalActions(index)
        if gameState.isLose() or gameState.isWin():
            return contestEvaluationFunction(gameState), None
        for action in legalMoves:
            state = gameState.generateSuccessor(index, action)
            if minMaxFlag:
                value = self.max_value(state, self.index, depth + 1, alpha, beta)[0]
            else:
                value = self.min_value(state, index + 1, depth, alpha, beta)[0]
            if v > value:
                v = value
                move = action
            if v < alpha:
                return v, move
            beta = min(beta, v)
        return v, move

def contestEvaluationFunction(currentGameState):
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()  # food available from current state
    currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    score = currentGameState.getScore()
    score += sum([1.0 / manhattanDistance(currentPos, position) for position in currentCapsules])
    score += sum([1.0 / manhattanDistance(currentPos, position) for position in currentFood.asList()])
    totaltime = sum([time for time in currentScaredTimes])
    score += totaltime
    if totaltime > 0 and currentPos == currentGhostStates[0].start.pos:
        score -= 10
    for ghostState in currentGhostStates:
        distance = manhattanDistance(currentPos, ghostState.getPosition())
        if not totaltime > 0:
            if distance < 3:
                score -= 100

    return score
