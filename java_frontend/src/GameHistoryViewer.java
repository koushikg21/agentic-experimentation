import javax.swing.*;
import java.awt.*;
import java.awt.geom.Path2D;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Minimal Swing front-end that visualizes the most recent tic-tac-toe series stored in game_history.csv.
 */
public class GameHistoryViewer extends JFrame {
    private final List<GameRecord> records;
    private final ScoreAnimationPanel scorePanel;
    private final MinimaxPanel minimaxPanel;
    private final JLabel statusLabel = new JLabel("Loading series...");
    private final JButton replayButton = new JButton("Replay Animation");
    private Timer animationTimer;
    private int displayedGames = 0;

    public GameHistoryViewer(List<GameRecord> records, Path csvPath) {
        super("Tic-Tac Toe Series Visualizer");
        this.records = records;
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        setPreferredSize(new Dimension(900, 700));

        scorePanel = new ScoreAnimationPanel(records);
        minimaxPanel = new MinimaxPanel(records);

        JPanel summaryPanel = buildSummaryPanel(records, csvPath);

        JPanel content = new JPanel(new BorderLayout(10, 10));
        content.setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));
        content.add(summaryPanel, BorderLayout.NORTH);
        content.add(scorePanel, BorderLayout.CENTER);
        content.add(minimaxPanel, BorderLayout.SOUTH);

        setContentPane(content);
        pack();
        setLocationRelativeTo(null);

        replayButton.addActionListener(e -> restartAnimation());
        restartAnimation();
    }

    private JPanel buildSummaryPanel(List<GameRecord> records, Path csvPath) {
        GameRecord first = records.get(0);
        JPanel summaryPanel = new JPanel();
        summaryPanel.setLayout(new BoxLayout(summaryPanel, BoxLayout.Y_AXIS));
        summaryPanel.setOpaque(false);

        JLabel title = new JLabel(
                String.format(Locale.US, "Series %s · %d games", first.seriesId, records.size()));
        title.setFont(title.getFont().deriveFont(Font.BOLD, 18f));
        summaryPanel.add(title);

        JLabel models = new JLabel(String.format(Locale.US, "%s (A) vs %s (B)", first.modelA, first.modelB));
        models.setFont(models.getFont().deriveFont(Font.PLAIN, 14f));
        summaryPanel.add(models);

        JLabel source = new JLabel("Source: " + csvPath.toAbsolutePath());
        source.setFont(source.getFont().deriveFont(Font.PLAIN, 12f));
        summaryPanel.add(source);

        JPanel bottomRow = new JPanel(new BorderLayout());
        bottomRow.setOpaque(false);
        statusLabel.setFont(statusLabel.getFont().deriveFont(Font.PLAIN, 13f));
        bottomRow.add(statusLabel, BorderLayout.CENTER);

        JPanel controls = new JPanel();
        controls.setOpaque(false);
        controls.add(replayButton);
        bottomRow.add(controls, BorderLayout.EAST);

        summaryPanel.add(Box.createVerticalStrut(6));
        summaryPanel.add(bottomRow);
        summaryPanel.add(Box.createVerticalStrut(10));
        summaryPanel.add(new JSeparator());
        summaryPanel.add(Box.createVerticalStrut(4));
        return summaryPanel;
    }

    private void restartAnimation() {
        displayedGames = 0;
        scorePanel.setDisplayedGames(displayedGames);
        minimaxPanel.setDisplayedGames(displayedGames);
        if (animationTimer != null) {
            animationTimer.stop();
        }
        animationTimer = new Timer(1200, e -> stepAnimation());
        animationTimer.setInitialDelay(400);
        animationTimer.start();
        statusLabel.setText("Animating " + records.size() + " games...");
    }

    private void stepAnimation() {
        if (displayedGames >= records.size()) {
            animationTimer.stop();
            statusLabel.setText("Animation complete. Click replay to watch again.");
            return;
        }
        displayedGames++;
        scorePanel.setDisplayedGames(displayedGames);
        minimaxPanel.setDisplayedGames(displayedGames);

        GameRecord current = records.get(displayedGames - 1);
        statusLabel.setText(String.format(Locale.US,
                "Game %d: winner %s · A pts %d (minimax %s) · B pts %d (minimax %s)",
                current.gameNumber,
                current.winner,
                current.aPoints,
                current.aUsedMinimax ? "yes" : "no",
                current.bPoints,
                current.bUsedMinimax ? "yes" : "no"));
    }

    public static void main(String[] args) {
        try {
            Path csvPath = args.length > 0 ? Paths.get(args[0]) : Paths.get("game_history.csv");
            if (!Files.exists(csvPath)) {
                System.err.println("Could not find " + csvPath.toAbsolutePath());
                return;
            }
            List<GameRecord> latest = CsvLoader.loadLatestSeries(csvPath);
            if (latest.isEmpty()) {
                System.err.println("No games detected in " + csvPath);
                return;
            }
            SwingUtilities.invokeLater(() -> new GameHistoryViewer(latest, csvPath).setVisible(true));
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    /**
     * Model of an individual game row.
     */
    private static class GameRecord {
        final LocalDateTime timestamp;
        final String seriesId;
        final int seriesGamesConfigured;
        final String modelA;
        final String modelB;
        final int gameNumber;
        final String winner;
        final String starter;
        final int moves;
        final boolean aUsedMinimax;
        final boolean bUsedMinimax;
        final int aTokens;
        final int bTokens;
        final int aPoints;
        final int bPoints;
        final String incidents;

        GameRecord(List<String> raw) {
            timestamp = CsvLoader.parseTimestamp(raw.get(0));
            seriesId = raw.get(1);
            seriesGamesConfigured = CsvLoader.parseInt(raw.get(2));
            modelA = raw.get(3);
            modelB = raw.get(4);
            gameNumber = CsvLoader.parseInt(raw.get(5));
            winner = raw.get(6);
            starter = raw.get(7);
            moves = CsvLoader.parseInt(raw.get(8));
            aUsedMinimax = CsvLoader.parseBoolean(raw.get(9));
            bUsedMinimax = CsvLoader.parseBoolean(raw.get(10));
            aTokens = CsvLoader.parseInt(raw.get(11));
            bTokens = CsvLoader.parseInt(raw.get(12));
            aPoints = CsvLoader.parseInt(raw.get(13));
            bPoints = CsvLoader.parseInt(raw.get(14));
            incidents = raw.size() > 15 ? raw.get(15) : "";
        }
    }

    /**
     * Responsible for parsing the CSV file and returning the most recent series.
     */
    private static class CsvLoader {
        private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS");

        static List<GameRecord> loadLatestSeries(Path csvPath) throws IOException {
            List<String> lines = Files.readAllLines(csvPath);
            List<GameRecord> allRecords = new ArrayList<>();
            for (int i = 1; i < lines.size(); i++) { // skip header
                String line = lines.get(i);
                if (line.trim().isEmpty()) {
                    continue;
                }
                List<String> cells = parseCsvRow(line);
                if (cells.size() < 15) {
                    continue; // skip malformed lines
                }
                allRecords.add(new GameRecord(cells));
            }
            if (allRecords.isEmpty()) {
                return List.of();
            }
            GameRecord latest = allRecords.stream()
                    .max(Comparator.comparing((GameRecord g) -> g.timestamp)
                            .thenComparing(g -> g.seriesId)
                            .thenComparingInt(g -> g.gameNumber))
                    .orElseThrow();
            List<GameRecord> latestSeries = new ArrayList<>();
            for (GameRecord record : allRecords) {
                if (Objects.equals(record.seriesId, latest.seriesId)) {
                    latestSeries.add(record);
                }
            }
            latestSeries.sort(Comparator.comparingInt(g -> g.gameNumber));
            return latestSeries;
        }

        private static List<String> parseCsvRow(String line) {
            List<String> cells = new ArrayList<>();
            StringBuilder sb = new StringBuilder();
            boolean inQuotes = false;
            for (int i = 0; i < line.length(); i++) {
                char c = line.charAt(i);
                if (c == '"') {
                    inQuotes = !inQuotes;
                } else if (c == ',' && !inQuotes) {
                    cells.add(sb.toString());
                    sb.setLength(0);
                } else {
                    sb.append(c);
                }
            }
            cells.add(sb.toString());
            return cells;
        }

        private static LocalDateTime parseTimestamp(String value) {
            try {
                return LocalDateTime.parse(value, FORMATTER);
            } catch (Exception ex) {
                return LocalDateTime.MIN;
            }
        }

        private static int parseInt(String value) {
            try {
                return Integer.parseInt(value.trim());
            } catch (NumberFormatException ex) {
                return 0;
            }
        }

        private static boolean parseBoolean(String value) {
            return Boolean.parseBoolean(value.trim());
        }
    }

    /**
     * Draws a simple line chart that animates score progression for players A and B.
     */
    private static class ScoreAnimationPanel extends JPanel {
        private static final Color COLOR_A = new Color(224, 98, 87);
        private static final Color COLOR_B = new Color(78, 129, 221);
        private final List<GameRecord> records;
        private final int maxPoints;
        private int displayedGames = 0;

        ScoreAnimationPanel(List<GameRecord> records) {
            this.records = records;
            this.maxPoints = records.stream()
                    .mapToInt(r -> Math.max(r.aPoints, r.bPoints))
                    .max()
                    .orElse(10);
            setPreferredSize(new Dimension(800, 420));
            setBackground(Color.WHITE);
        }

        void setDisplayedGames(int displayedGames) {
            this.displayedGames = displayedGames;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int width = getWidth();
            int height = getHeight();
            int left = 60;
            int right = width - 20;
            int top = 30;
            int bottom = height - 50;

            g2.setColor(new Color(245, 245, 245));
            g2.fillRoundRect(left - 40, top - 20, right - left + 60, bottom - top + 70, 20, 20);

            g2.setColor(Color.DARK_GRAY);
            g2.drawLine(left, bottom, right, bottom);
            g2.drawLine(left, bottom, left, top);

            g2.setFont(g2.getFont().deriveFont(Font.PLAIN, 11f));
            g2.drawString("Games", (right + left) / 2 - 15, bottom + 30);
            g2.drawString("Points", left - 50, top - 10);

            if (records.isEmpty() || displayedGames == 0) {
                g2.drawString("Replay the animation to see the score progression.", left + 20, (top + bottom) / 2);
                g2.dispose();
                return;
            }

            int totalGames = records.size();
            double xStep = totalGames > 1 ? (double) (right - left) / (totalGames - 1) : 0;
            // grid lines
            g2.setColor(new Color(220, 220, 220));
            for (int i = 0; i < displayedGames; i++) {
                int x = (int) Math.round(left + i * xStep);
                g2.drawLine(x, bottom - 4, x, top);
            }

            drawSeries(g2, left, bottom, xStep, totalGames, COLOR_A, true);
            drawSeries(g2, left, bottom, xStep, totalGames, COLOR_B, false);

            g2.dispose();
        }

        private void drawSeries(Graphics2D g2, int left, int bottom, double xStep, int totalGames,
                                 Color color, boolean forPlayerA) {
            if (displayedGames == 0) {
                return;
            }
            Path2D path = new Path2D.Double();
            for (int i = 0; i < displayedGames; i++) {
                GameRecord record = records.get(i);
                double value = forPlayerA ? record.aPoints : record.bPoints;
                double normalized = value / Math.max(1, maxPoints);
                double x = left + (totalGames == 1 ? 0 : i * xStep);
                double y = bottom - normalized * (bottom - 60);
                if (i == 0) {
                    path.moveTo(x, y);
                } else {
                    path.lineTo(x, y);
                }
            }
            g2.setStroke(new BasicStroke(3f));
            g2.setColor(color);
            g2.draw(path);

            g2.setStroke(new BasicStroke(1f));
            for (int i = 0; i < displayedGames; i++) {
                GameRecord record = records.get(i);
                double value = forPlayerA ? record.aPoints : record.bPoints;
                double normalized = value / Math.max(1, maxPoints);
                int x = (int) Math.round(left + (totalGames == 1 ? 0 : i * xStep));
                int y = (int) Math.round(bottom - normalized * (bottom - 60));
                g2.setColor(Color.WHITE);
                g2.fillOval(x - 6, y - 6, 12, 12);
                g2.setColor(color);
                g2.drawOval(x - 6, y - 6, 12, 12);
                g2.setFont(g2.getFont().deriveFont(Font.BOLD, 10f));
                g2.drawString(String.valueOf((int) value), x - 8, y - 10 - (forPlayerA ? 0 : 5));
            }

            String label = forPlayerA ? "Player A" : "Player B";
            g2.setColor(color);
            g2.fillRect(left + 10 + (forPlayerA ? 0 : 120), 10, 15, 15);
            g2.setColor(Color.DARK_GRAY);
            g2.drawString(label, left + 30 + (forPlayerA ? 0 : 120), 22);
        }
    }

    /**
     * Visualizes when minimax was used in the latest series.
     */
    private static class MinimaxPanel extends JPanel {
        private final List<GameRecord> records;
        private int displayedGames = 0;

        MinimaxPanel(List<GameRecord> records) {
            this.records = records;
            setPreferredSize(new Dimension(800, 180));
            setBackground(new Color(252, 252, 252));
        }

        void setDisplayedGames(int displayedGames) {
            this.displayedGames = displayedGames;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            g2.setColor(Color.DARK_GRAY);
            g2.setFont(g2.getFont().deriveFont(Font.BOLD, 14f));
            g2.drawString("Minimax usage timeline", 12, 20);

            if (records.isEmpty()) {
                g2.dispose();
                return;
            }

            int width = getWidth();
            int left = 40;
            int right = width - 40;
            int top = 40;
            int boxWidth = Math.max(20, (right - left) / records.size() - 4);
            int spacing = 4;

            for (int i = 0; i < records.size(); i++) {
                int x = left + i * (boxWidth + spacing);
                GameRecord record = records.get(i);
                Color outline = i < displayedGames ? Color.GRAY : new Color(210, 210, 210);
                g2.setColor(outline);
                g2.fillRoundRect(x, top, boxWidth, 80, 8, 8);

                if (i < displayedGames) {
                    if (record.aUsedMinimax) {
                        g2.setColor(new Color(224, 98, 87, 200));
                        g2.fillRoundRect(x + 2, top + 5, boxWidth - 4, 30, 6, 6);
                        g2.setColor(Color.WHITE);
                        g2.drawString("A", x + boxWidth / 2 - 4, top + 25);
                    }
                    if (record.bUsedMinimax) {
                        g2.setColor(new Color(78, 129, 221, 200));
                        g2.fillRoundRect(x + 2, top + 45, boxWidth - 4, 30, 6, 6);
                        g2.setColor(Color.WHITE);
                        g2.drawString("B", x + boxWidth / 2 - 4, top + 65);
                    }
                    if (!record.aUsedMinimax && !record.bUsedMinimax) {
                        g2.setColor(new Color(120, 120, 120));
                        g2.drawString("—", x + boxWidth / 2 - 3, top + 52);
                    }
                }

                g2.setColor(Color.DARK_GRAY);
                g2.setFont(g2.getFont().deriveFont(Font.PLAIN, 11f));
                g2.drawString("G" + record.gameNumber, x + 4, top + 100);
                g2.drawString(record.winner, x + 4, top + 115);
            }

            g2.dispose();
        }
    }
}
