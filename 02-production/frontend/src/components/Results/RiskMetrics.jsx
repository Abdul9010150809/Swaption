import React from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Tag } from 'antd';
import {
  DollarOutlined,
  PercentageOutlined,
  BarChartOutlined,
  WarningOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { formatCurrency, formatPercentage, formatNumber } from '../../utils/formatters';

/**
 * Risk Metrics Display Component
 * Shows comprehensive risk analysis for pricing results
 */
const RiskMetrics = ({ riskData, loading = false }) => {
  if (!riskData || loading) {
    return (
      <Card title="Risk Metrics" loading={loading}>
        <div style={{ textAlign: 'center', padding: '20px' }}>
          No risk data available
        </div>
      </Card>
    );
  }

  const {
    var: varValue,
    cvar: cvarValue,
    confidence_level: confidenceLevel,
    time_horizon: timeHorizon,
    method: method,
    greeks,
    stress_test_results: stressResults,
    max_drawdown: maxDrawdown
  } = riskData;

  // VaR/CVaR Card
  const renderVaRCard = () => (
    <Card title="Value at Risk (VaR)" size="small">
      <Row gutter={16}>
        <Col span={12}>
          <Statistic
            title={`VaR (${(confidenceLevel * 100).toFixed(0)}% Confidence)`}
            value={varValue}
            prefix={<DollarOutlined />}
            valueStyle={{ color: '#cf1322' }}
            formatter={(value) => formatCurrency(Math.abs(value), 'USD', 2)}
          />
        </Col>
        <Col span={12}>
          <Statistic
            title="Conditional VaR (CVaR)"
            value={cvarValue}
            prefix={<DollarOutlined />}
            valueStyle={{ color: '#d4380d' }}
            formatter={(value) => formatCurrency(Math.abs(value), 'USD', 2)}
          />
        </Col>
      </Row>
      <div style={{ marginTop: 16 }}>
        <Tag color="orange">Method: {method}</Tag>
        <Tag color="blue">Time Horizon: {timeHorizon} day(s)</Tag>
      </div>
    </Card>
  );

  // Greeks Card
  const renderGreeksCard = () => {
    if (!greeks) return null;

    const greekColumns = [
      {
        title: 'Greek',
        dataIndex: 'name',
        key: 'name',
        width: 80
      },
      {
        title: 'Value',
        dataIndex: 'value',
        key: 'value',
        render: (value, record) => {
          const color = value >= 0 ? '#3f8600' : '#cf1322';
          return (
            <span style={{ color }}>
              {record.name === 'gamma' ? value.toFixed(6) :
               record.name === 'delta' ? value.toFixed(4) :
               value.toFixed(4)}
            </span>
          );
        }
      },
      {
        title: 'Risk Contribution',
        dataIndex: 'risk',
        key: 'risk',
        render: (value) => (
          <span style={{ color: value > 0.01 ? '#cf1322' : '#3f8600' }}>
            {formatCurrency(value, 'USD', 4)}
          </span>
        )
      }
    ];

    const greekData = [
      {
        key: 'delta',
        name: 'Delta',
        value: greeks.delta || 0,
        risk: greeks.delta_risk || 0
      },
      {
        key: 'gamma',
        name: 'Gamma',
        value: greeks.gamma || 0,
        risk: greeks.gamma_risk || 0
      },
      {
        key: 'theta',
        name: 'Theta',
        value: greeks.theta || 0,
        risk: greeks.theta_risk || 0
      },
      {
        key: 'vega',
        name: 'Vega',
        value: greeks.vega || 0,
        risk: greeks.vega_risk || 0
      },
      {
        key: 'rho',
        name: 'Rho',
        value: greeks.rho || 0,
        risk: greeks.rho_risk || 0
      }
    ];

    return (
      <Card title="Option Greeks" size="small">
        <Table
          columns={greekColumns}
          dataSource={greekData}
          pagination={false}
          size="small"
          showHeader={true}
        />
        <div style={{ marginTop: 16 }}>
          <Tag color="green">
            Total Greeks Risk: {formatCurrency(greeks.total_greeks_risk || 0, 'USD', 4)}
          </Tag>
        </div>
      </Card>
    );
  };

  // Stress Test Results Card
  const renderStressTestCard = () => {
    if (!stressResults) return null;

    const stressColumns = [
      {
        title: 'Scenario',
        dataIndex: 'scenario',
        key: 'scenario',
        width: 150
      },
      {
        title: 'Loss Amount',
        dataIndex: 'loss',
        key: 'loss',
        render: (value) => formatCurrency(value, 'USD', 2),
        sorter: (a, b) => a.loss - b.loss
      },
      {
        title: 'Loss %',
        dataIndex: 'loss_percentage',
        key: 'loss_percentage',
        render: (value) => (
          <span style={{ color: value > 5 ? '#cf1322' : value > 2 ? '#fa8c16' : '#3f8600' }}>
            {formatPercentage(value / 100, 2)}
          </span>
        ),
        sorter: (a, b) => a.loss_percentage - b.loss_percentage
      },
      {
        title: 'Risk Level',
        key: 'risk_level',
        render: (_, record) => {
          const lossPct = record.loss_percentage;
          if (lossPct > 10) return <Tag color="red">Critical</Tag>;
          if (lossPct > 5) return <Tag color="orange">High</Tag>;
          if (lossPct > 2) return <Tag color="yellow">Medium</Tag>;
          return <Tag color="green">Low</Tag>;
        }
      }
    ];

    const stressData = Object.entries(stressResults).map(([scenario, data]) => ({
      key: scenario,
      scenario: scenario.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      loss: data.loss,
      loss_percentage: data.loss_percentage
    }));

    return (
      <Card title="Stress Test Results" size="small">
        <Table
          columns={stressColumns}
          dataSource={stressData}
          pagination={false}
          size="small"
          scroll={{ y: 200 }}
        />
      </Card>
    );
  };

  // Drawdown Analysis Card
  const renderDrawdownCard = () => {
    if (!maxDrawdown) return null;

    const drawdownPercentage = (maxDrawdown.max_drawdown * 100);

    return (
      <Card title="Drawdown Analysis" size="small">
        <Row gutter={16}>
          <Col span={12}>
            <Statistic
              title="Maximum Drawdown"
              value={drawdownPercentage}
              suffix="%"
              valueStyle={{ color: drawdownPercentage > 20 ? '#cf1322' : '#3f8600' }}
              formatter={(value) => formatPercentage(value / 100, 2)}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="Current Drawdown"
              value={maxDrawdown.current_drawdown * 100}
              suffix="%"
              formatter={(value) => formatPercentage(value / 100, 2)}
            />
          </Col>
        </Row>
        <div style={{ marginTop: 16 }}>
          <Progress
            percent={Math.abs(drawdownPercentage)}
            status={drawdownPercentage > 20 ? 'exception' : 'normal'}
            strokeColor={drawdownPercentage > 20 ? '#cf1322' : '#3f8600'}
            format={() => `${Math.abs(drawdownPercentage).toFixed(1)}%`}
          />
        </div>
      </Card>
    );
  };

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          {renderVaRCard()}
        </Col>
        <Col xs={24} lg={12}>
          {renderDrawdownCard()}
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          {renderGreeksCard()}
        </Col>
        <Col xs={24} lg={12}>
          {renderStressTestCard()}
        </Col>
      </Row>
    </div>
  );
};

export default RiskMetrics;